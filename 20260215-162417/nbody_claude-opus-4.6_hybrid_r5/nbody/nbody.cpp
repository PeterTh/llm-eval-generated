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
static constexpr int BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

// Tiled force computation kernel using shared memory
__global__ void computeForcesKernel(
    const double* __restrict__ all_px,
    const double* __restrict__ all_py,
    const double* __restrict__ all_pz,
    double* __restrict__ vx,
    double* __restrict__ vy,
    double* __restrict__ vz,
    int localStart, int localCount, int N)
{
    __shared__ double sh_px[BLOCK_SIZE];
    __shared__ double sh_py[BLOCK_SIZE];
    __shared__ double sh_pz[BLOCK_SIZE];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double myPx = 0.0, myPy = 0.0, myPz = 0.0;

    if (idx < localCount) {
        int i = localStart + idx;
        myPx = all_px[i];
        myPy = all_py[i];
        myPz = all_pz[i];
    }

    int numTiles = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int tile = 0; tile < numTiles; ++tile) {
        int jIdx = tile * BLOCK_SIZE + threadIdx.x;
        sh_px[threadIdx.x] = (jIdx < N) ? all_px[jIdx] : 0.0;
        sh_py[threadIdx.x] = (jIdx < N) ? all_py[jIdx] : 0.0;
        sh_pz[threadIdx.x] = (jIdx < N) ? all_pz[jIdx] : 0.0;
        __syncthreads();

        if (idx < localCount) {
            int jEnd = min(BLOCK_SIZE, N - tile * BLOCK_SIZE);
            #pragma unroll 8
            for (int k = 0; k < jEnd; ++k) {
                double dx = sh_px[k] - myPx;
                double dy = sh_py[k] - myPy;
                double dz = sh_pz[k] - myPz;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = rsqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (idx < localCount) {
        vx[idx] += DT * Fx;
        vy[idx] += DT * Fy;
        vz[idx] += DT * Fz;
    }
}

__global__ void integrateKernel(
    double* __restrict__ px,
    double* __restrict__ py,
    double* __restrict__ pz,
    const double* __restrict__ vx,
    const double* __restrict__ vy,
    const double* __restrict__ vz,
    int localCount)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= localCount) return;
    px[idx] += vx[idx] * DT;
    py[idx] += vy[idx] * DT;
    pz[idx] += vz[idx] * DT;
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

double computeTotalEnergy(const double* px, const double* py, const double* pz,
                          const double* vx, const double* vy, const double* vz, int N) {
    double kinetic = 0.0;
    #pragma omp parallel for reduction(+:kinetic)
    for (int i = 0; i < N; ++i) {
        kinetic += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    double potential = 0.0;
    #pragma omp parallel for reduction(+:potential)
    for (int i = 0; i < N; ++i) {
        for (int j = i + 1; j < N; ++j) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
    }
    return kinetic + potential;
}

bool validateSimulation(const double* px, const double* py, const double* pz,
                        const double* vx, const double* vy, const double* vz, int N) {
    bool valid = true;
    #pragma omp parallel for
    for (int i = 0; i < N; ++i) {
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || !std::isfinite(pz[i]) ||
            !std::isfinite(vx[i]) || !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            #pragma omp atomic write
            valid = false;
        }
        const double maxVal = 1e6;
        if (std::abs(px[i]) > maxVal || std::abs(py[i]) > maxVal || std::abs(pz[i]) > maxVal ||
            std::abs(vx[i]) > maxVal || std::abs(vy[i]) > maxVal || std::abs(vz[i]) > maxVal) {
            #pragma omp atomic write
            valid = false;
        }
    }
    if (!valid) {
        printf("Validation failed: found NaN, Inf, or extreme values\n");
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

    // Assign GPU per MPI rank
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    CUDA_CHECK(cudaSetDevice(rank % numDevices));

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d, GPUs: %d, OpenMP threads: %d\n",
               numProcs, numDevices, omp_get_max_threads());
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int N = numBodies;

    // SoA host arrays for all bodies
    std::vector<double> all_px(N), all_py(N), all_pz(N);
    std::vector<double> all_vx(N), all_vy(N), all_vz(N);

    // Initialize on rank 0, then broadcast
    if (rank == 0) {
        std::vector<Body> bodies(N);
        randomizeBodies(bodies);
        #pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            all_px[i] = bodies[i].pos.x;
            all_py[i] = bodies[i].pos.y;
            all_pz[i] = bodies[i].pos.z;
            all_vx[i] = bodies[i].vel.x;
            all_vy[i] = bodies[i].vel.y;
            all_vz[i] = bodies[i].vel.z;
        }
    }

    MPI_Bcast(all_px.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_py.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_pz.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_vx.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_vy.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(all_vz.data(), N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Partition bodies across MPI ranks
    int baseCount = N / numProcs;
    int remainder = N % numProcs;
    int localCount = baseCount + (rank < remainder ? 1 : 0);
    int localStart = rank * baseCount + std::min(rank, remainder);

    std::vector<int> recvcounts(numProcs), displs(numProcs);
    for (int r = 0; r < numProcs; ++r) {
        recvcounts[r] = baseCount + (r < remainder ? 1 : 0);
        displs[r] = r * baseCount + std::min(r, remainder);
    }

    // Allocate GPU memory
    double *d_all_px, *d_all_py, *d_all_pz;
    double *d_vx, *d_vy, *d_vz;
    CUDA_CHECK(cudaMalloc(&d_all_px, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_py, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_all_pz, N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vx, localCount * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vy, localCount * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vz, localCount * sizeof(double)));

    // Copy initial data to GPU
    CUDA_CHECK(cudaMemcpy(d_all_px, all_px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_py, all_py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_all_pz, all_pz.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, all_vx.data() + localStart, localCount * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, all_vy.data() + localStart, localCount * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, all_vz.data() + localStart, localCount * sizeof(double), cudaMemcpyHostToDevice));

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    int gridSize = (localCount + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int step = 0; step < numSteps; ++step) {
        // Force computation on GPU (O(N²) hot loop)
        computeForcesKernel<<<gridSize, BLOCK_SIZE>>>(
            d_all_px, d_all_py, d_all_pz,
            d_vx, d_vy, d_vz,
            localStart, localCount, N);

        // Position integration on GPU
        integrateKernel<<<gridSize, BLOCK_SIZE>>>(
            d_all_px + localStart, d_all_py + localStart, d_all_pz + localStart,
            d_vx, d_vy, d_vz,
            localCount);

        CUDA_CHECK(cudaDeviceSynchronize());

        // Copy local positions from GPU to host for MPI communication
        CUDA_CHECK(cudaMemcpy(all_px.data() + localStart, d_all_px + localStart,
                              localCount * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(all_py.data() + localStart, d_all_py + localStart,
                              localCount * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(all_pz.data() + localStart, d_all_pz + localStart,
                              localCount * sizeof(double), cudaMemcpyDeviceToHost));

        // Synchronize all positions across MPI ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       all_px.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       all_py.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       all_pz.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        // Copy updated positions back to GPU
        CUDA_CHECK(cudaMemcpy(d_all_px, all_px.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_all_py, all_py.data(), N * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_all_pz, all_pz.data(), N * sizeof(double), cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    // Copy local velocities from GPU for output/validation
    CUDA_CHECK(cudaMemcpy(all_vx.data() + localStart, d_vx,
                          localCount * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(all_vy.data() + localStart, d_vy,
                          localCount * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(all_vz.data() + localStart, d_vz,
                          localCount * sizeof(double), cudaMemcpyDeviceToHost));

    // Gather all velocities on rank 0
    if (numProcs > 1) {
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : all_vx.data() + localStart,
                    localCount, MPI_DOUBLE,
                    all_vx.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : all_vy.data() + localStart,
                    localCount, MPI_DOUBLE,
                    all_vy.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(rank == 0 ? MPI_IN_PLACE : all_vz.data() + localStart,
                    localCount, MPI_DOUBLE,
                    all_vz.data(), recvcounts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());

        if (printResults) {
            std::vector<double> bodyData;
            bodyData.reserve(N * 6);
            for (int i = 0; i < N; ++i) {
                bodyData.push_back(all_px[i]);
                bodyData.push_back(all_py[i]);
                bodyData.push_back(all_pz[i]);
                bodyData.push_back(all_vx[i]);
                bodyData.push_back(all_vy[i]);
                bodyData.push_back(all_vz[i]);
            }
            print_results(bodyData, "Bodies");
        }

        if (validate) {
            printf("Validating simulation results...\n");
            if (validateSimulation(all_px.data(), all_py.data(), all_pz.data(),
                                   all_vx.data(), all_vy.data(), all_vz.data(), N)) {
                double finalEnergy = computeTotalEnergy(
                    all_px.data(), all_py.data(), all_pz.data(),
                    all_vx.data(), all_vy.data(), all_vz.data(), N);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                cudaFree(d_all_px); cudaFree(d_all_py); cudaFree(d_all_pz);
                cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
                MPI_Finalize();
                return 1;
            }
        }
    }

    cudaFree(d_all_px); cudaFree(d_all_py); cudaFree(d_all_pz);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);

    MPI_Finalize();
    return 0;
}
