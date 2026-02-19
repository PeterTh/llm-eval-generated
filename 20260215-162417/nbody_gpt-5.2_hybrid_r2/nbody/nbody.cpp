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

#define CUDA_CHECK(call)                                                                    \
    do {                                                                                    \
        cudaError_t _e = (call);                                                            \
        if (_e != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(_e)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                    \
        }                                                                                   \
    } while (0)

static inline void randomizeBodiesAoS(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

__global__ void computeForcesKernel(const double* __restrict__ globalPosXYZ,
                                   const double* __restrict__ localPosXYZ,
                                   double* __restrict__ localVelXYZ,
                                   int n,
                                   int localN) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= localN) return;

    const double ix = localPosXYZ[3 * i + 0];
    const double iy = localPosXYZ[3 * i + 1];
    const double iz = localPosXYZ[3 * i + 2];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    extern __shared__ double shmem[];
    double* shx = shmem;
    double* shy = shmem + blockDim.x;
    double* shz = shmem + 2 * blockDim.x;

    for (int tile = 0; tile < n; tile += blockDim.x) {
        const int j = tile + threadIdx.x;
        if (j < n) {
            shx[threadIdx.x] = globalPosXYZ[3 * j + 0];
            shy[threadIdx.x] = globalPosXYZ[3 * j + 1];
            shz[threadIdx.x] = globalPosXYZ[3 * j + 2];
        }
        __syncthreads();

        const int tileSize = (n - tile) < (int)blockDim.x ? (n - tile) : (int)blockDim.x;
        for (int k = 0; k < tileSize; ++k) {
            const double dx = shx[k] - ix;
            const double dy = shy[k] - iy;
            const double dz = shz[k] - iz;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    localVelXYZ[3 * i + 0] += DT * Fx;
    localVelXYZ[3 * i + 1] += DT * Fy;
    localVelXYZ[3 * i + 2] += DT * Fz;
}

__global__ void integrateKernel(double* __restrict__ localPosXYZ,
                               const double* __restrict__ localVelXYZ,
                               int localN) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= localN) return;

    localPosXYZ[3 * i + 0] += localVelXYZ[3 * i + 0] * DT;
    localPosXYZ[3 * i + 1] += localVelXYZ[3 * i + 1] * DT;
    localPosXYZ[3 * i + 2] += localVelXYZ[3 * i + 2] * DT;
}

static double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

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

static bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

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

static void printUsage(const char* progName) {
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
        printf("MPI ranks: %d\n", size);
        printf("OpenMP threads/rank: %d\n", omp_get_max_threads());
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount <= 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices found.\n");
        MPI_Finalize();
        return 1;
    }
    CUDA_CHECK(cudaSetDevice(rank % deviceCount));

    const int n = numBodies;
    const int base = n / size;
    const int rem = n % size;
    const int localN = base + (rank < rem ? 1 : 0);

    std::vector<int> countsBodies(size), displsBodies(size);
    std::vector<int> countsXYZ(size), displsXYZ(size);
    int disp = 0;
    for (int r = 0; r < size; ++r) {
        const int c = base + (r < rem ? 1 : 0);
        countsBodies[r] = c;
        displsBodies[r] = disp;
        countsXYZ[r] = 3 * c;
        displsXYZ[r] = 3 * disp;
        disp += c;
    }

    // Host pinned buffers for local state and global positions.
    double* h_localPosXYZ = nullptr;
    double* h_localVelXYZ = nullptr;
    double* h_globalPosXYZ = nullptr;
    CUDA_CHECK(cudaHostAlloc((void**)&h_localPosXYZ, sizeof(double) * 3 * localN, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_localVelXYZ, sizeof(double) * 3 * localN, cudaHostAllocDefault));
    CUDA_CHECK(cudaHostAlloc((void**)&h_globalPosXYZ, sizeof(double) * 3 * n, cudaHostAllocDefault));

    std::vector<double> initPosXYZ;
    std::vector<double> initVelXYZ;
    if (rank == 0) {
        std::vector<Body> initBodies(n);
        randomizeBodiesAoS(initBodies);
        initPosXYZ.resize(3 * n);
        initVelXYZ.resize(3 * n);
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            initPosXYZ[3 * i + 0] = initBodies[i].pos.x;
            initPosXYZ[3 * i + 1] = initBodies[i].pos.y;
            initPosXYZ[3 * i + 2] = initBodies[i].pos.z;
            initVelXYZ[3 * i + 0] = initBodies[i].vel.x;
            initVelXYZ[3 * i + 1] = initBodies[i].vel.y;
            initVelXYZ[3 * i + 2] = initBodies[i].vel.z;
        }
    }

    MPI_Scatterv(rank == 0 ? initPosXYZ.data() : nullptr,
                 countsXYZ.data(), displsXYZ.data(), MPI_DOUBLE,
                 h_localPosXYZ, 3 * localN, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    MPI_Scatterv(rank == 0 ? initVelXYZ.data() : nullptr,
                 countsXYZ.data(), displsXYZ.data(), MPI_DOUBLE,
                 h_localVelXYZ, 3 * localN, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    double* d_globalPosXYZ = nullptr;
    double* d_localPosXYZ = nullptr;
    double* d_localVelXYZ = nullptr;
    CUDA_CHECK(cudaMalloc((void**)&d_globalPosXYZ, sizeof(double) * 3 * n));
    CUDA_CHECK(cudaMalloc((void**)&d_localPosXYZ, sizeof(double) * 3 * localN));
    CUDA_CHECK(cudaMalloc((void**)&d_localVelXYZ, sizeof(double) * 3 * localN));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    CUDA_CHECK(cudaMemcpyAsync(d_localPosXYZ, h_localPosXYZ, sizeof(double) * 3 * localN, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_localVelXYZ, h_localVelXYZ, sizeof(double) * 3 * localN, cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    MPI_Barrier(MPI_COMM_WORLD);
    const double t0 = MPI_Wtime();

    const int block = 256;
    const int grid = (localN + block - 1) / block;
    const size_t shmemBytes = sizeof(double) * 3 * block;

    for (int step = 0; step < numSteps; ++step) {
        MPI_Allgatherv(h_localPosXYZ, 3 * localN, MPI_DOUBLE,
                       h_globalPosXYZ, countsXYZ.data(), displsXYZ.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        CUDA_CHECK(cudaMemcpyAsync(d_globalPosXYZ, h_globalPosXYZ, sizeof(double) * 3 * n, cudaMemcpyHostToDevice, stream));

        computeForcesKernel<<<grid, block, shmemBytes, stream>>>(d_globalPosXYZ, d_localPosXYZ, d_localVelXYZ, n, localN);
        integrateKernel<<<grid, block, 0, stream>>>(d_localPosXYZ, d_localVelXYZ, localN);

        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpyAsync(h_localPosXYZ, d_localPosXYZ, sizeof(double) * 3 * localN, cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double t1 = MPI_Wtime();
    const double localSec = t1 - t0;
    double maxSec = 0.0;
    MPI_Reduce(&localSec, &maxSec, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", (long)(maxSec * 1000.0));
    }

    // Final gather for output/validation.
    CUDA_CHECK(cudaMemcpyAsync(h_localVelXYZ, d_localVelXYZ, sizeof(double) * 3 * localN, cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<double> finalPosXYZ;
    std::vector<double> finalVelXYZ;
    if (rank == 0) {
        finalPosXYZ.resize(3 * n);
        finalVelXYZ.resize(3 * n);
    }

    MPI_Gatherv(h_localPosXYZ, 3 * localN, MPI_DOUBLE,
                rank == 0 ? finalPosXYZ.data() : nullptr,
                countsXYZ.data(), displsXYZ.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    MPI_Gatherv(h_localVelXYZ, 3 * localN, MPI_DOUBLE,
                rank == 0 ? finalVelXYZ.data() : nullptr,
                countsXYZ.data(), displsXYZ.data(), MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0 && (printResults || validate)) {
        std::vector<Body> bodies(n);
#pragma omp parallel for
        for (int i = 0; i < n; ++i) {
            bodies[i].pos.x = finalPosXYZ[3 * i + 0];
            bodies[i].pos.y = finalPosXYZ[3 * i + 1];
            bodies[i].pos.z = finalPosXYZ[3 * i + 2];
            bodies[i].vel.x = finalVelXYZ[3 * i + 0];
            bodies[i].vel.y = finalVelXYZ[3 * i + 1];
            bodies[i].vel.z = finalVelXYZ[3 * i + 2];
        }

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.resize((size_t)n * 6);
#pragma omp parallel for
            for (int i = 0; i < n; ++i) {
                bodyData[6 * i + 0] = bodies[i].pos.x;
                bodyData[6 * i + 1] = bodies[i].pos.y;
                bodyData[6 * i + 2] = bodies[i].pos.z;
                bodyData[6 * i + 3] = bodies[i].vel.x;
                bodyData[6 * i + 4] = bodies[i].vel.y;
                bodyData[6 * i + 5] = bodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                CUDA_CHECK(cudaStreamDestroy(stream));
                CUDA_CHECK(cudaFree(d_globalPosXYZ));
                CUDA_CHECK(cudaFree(d_localPosXYZ));
                CUDA_CHECK(cudaFree(d_localVelXYZ));
                CUDA_CHECK(cudaFreeHost(h_localPosXYZ));
                CUDA_CHECK(cudaFreeHost(h_localVelXYZ));
                CUDA_CHECK(cudaFreeHost(h_globalPosXYZ));
                MPI_Finalize();
                return 1;
            }
        }
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUDA_CHECK(cudaFree(d_globalPosXYZ));
    CUDA_CHECK(cudaFree(d_localPosXYZ));
    CUDA_CHECK(cudaFree(d_localVelXYZ));
    CUDA_CHECK(cudaFreeHost(h_localPosXYZ));
    CUDA_CHECK(cudaFreeHost(h_localVelXYZ));
    CUDA_CHECK(cudaFreeHost(h_globalPosXYZ));

    MPI_Finalize();
    return 0;
}
