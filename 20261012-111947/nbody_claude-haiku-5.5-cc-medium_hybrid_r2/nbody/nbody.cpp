#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int THREADS_PER_BLOCK = 64;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

#define CUDA_CHECK(call)                                                                        \
    do {                                                                                        \
        const cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                              \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err_)); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                       \
        }                                                                                       \
    } while (0)

void randomizeBodies(std::vector<Vec3>& pos, std::vector<Vec3>& vel, unsigned int seed = 42) {
    for (size_t i = 0; i < pos.size(); ++i) {
        pos[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pos[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel[i].x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel[i].y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vel[i].z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// One thread per body in [firstBody, firstBody + count). Each thread accumulates over all n bodies
// in index order, exactly as the serial loop did. Source positions are staged through shared memory.
__global__ void computeForcesKernel(const Vec3* __restrict__ pos, Vec3* __restrict__ vel,
                                    int firstBody, int count, int n) {
    __shared__ Vec3 tile[THREADS_PER_BLOCK];
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = li < count;
    const Vec3 pi = active ? pos[firstBody + li] : Vec3();
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += THREADS_PER_BLOCK) {
        const int j = base + threadIdx.x;
        if (j < n) tile[threadIdx.x] = pos[j];
        __syncthreads();

        const int tileCount = (n - base < THREADS_PER_BLOCK) ? n - base : THREADS_PER_BLOCK;
        for (int k = 0; k < tileCount; ++k) {
            const Vec3 pj = tile[k];
            const double dx = pj.x - pi.x;
            const double dy = pj.y - pi.y;
            const double dz = pj.z - pi.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    if (active) {
        vel[li].x += DT * Fx;
        vel[li].y += DT * Fy;
        vel[li].z += DT * Fz;
    }
}

__global__ void integrateKernel(Vec3* __restrict__ pos, const Vec3* __restrict__ vel, int firstBody, int count) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= count) return;
    Vec3& p = pos[firstBody + li];
    const Vec3 v = vel[li];
    p.x += v.x * DT;
    p.y += v.y * DT;
    p.z += v.z * DT;
}

double computeTotalEnergy(const std::vector<Vec3>& pos, const std::vector<Vec3>& vel) {
    double energy = 0.0;
    const size_t n = pos.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& v : vel) {
        energy += 0.5 * (v.x * v.x + v.y * v.y + v.z * v.z);
    }

    // Potential energy (assuming unit mass for all bodies). Each row i is independent, so rows are
    // summed in parallel and then accumulated in index order.
    std::vector<double> rowPotential(n, 0.0);
    #pragma omp parallel for schedule(dynamic, 64)
    for (size_t i = 0; i < n; ++i) {
        double row = 0.0;
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = pos[j].x - pos[i].x;
            const double dy = pos[j].y - pos[i].y;
            const double dz = pos[j].z - pos[i].z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            row -= 1.0 / dist;
        }
        rowPotential[i] = row;
    }
    for (size_t i = 0; i < n; ++i) {
        energy += rowPotential[i];
    }

    return energy;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Vec3>& pos, const std::vector<Vec3>& vel) {
    for (size_t i = 0; i < pos.size(); ++i) {
        // Check for NaN or Inf values
        if (!std::isfinite(pos[i].x) || !std::isfinite(pos[i].y) || !std::isfinite(pos[i].z) ||
            !std::isfinite(vel[i].x) || !std::isfinite(vel[i].y) || !std::isfinite(vel[i].z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(pos[i].x) > maxPos || std::abs(pos[i].y) > maxPos || std::abs(pos[i].z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vel[i].x) > maxVel || std::abs(vel[i].y) > maxVel || std::abs(vel[i].z) > maxVel) {
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

int runBenchmark(int argc, char** argv, int rank, int size) {
    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
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
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // MPI level: contiguous block of bodies per rank
    const int n = numBodies;
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const long long begin = static_cast<long long>(n) * r / size;
        const long long end = static_cast<long long>(n) * (r + 1) / size;
        displs[r] = static_cast<int>(begin);
        counts[r] = static_cast<int>(end - begin);
    }
    const int rankLo = displs[rank];
    const int rankN = counts[rank];

    MPI_Datatype vec3Type;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);

    // GPU level: each rank drives the GPUs of its node that are assigned to it (round-robin by
    // node-local rank), with one OpenMP thread per GPU
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int nodeRank = 0, nodeSize = 1;
    MPI_Comm_rank(nodeComm, &nodeRank);
    MPI_Comm_size(nodeComm, &nodeSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    std::vector<int> devices;
    for (int d = nodeRank; d < numDevices; d += nodeSize) {
        devices.push_back(d);
    }
    if (devices.empty()) {
        fprintf(stderr, "Rank %d has no GPU: %d GPUs on node, %d ranks on node\n", rank, numDevices, nodeSize);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int numGpus = static_cast<int>(devices.size());

    // Host state (every rank holds the full state; positions are exchanged each step)
    std::vector<Vec3> pos(n), vel(n);
    randomizeBodies(pos, vel);

    // Per-GPU sub-blocks of this rank's bodies, and device buffers
    std::vector<int> subBegin(numGpus), subN(numGpus);
    std::vector<Vec3*> dPos(numGpus, nullptr), dVel(numGpus, nullptr);
    for (int t = 0; t < numGpus; ++t) {
        subBegin[t] = rankLo + static_cast<int>(static_cast<long long>(rankN) * t / numGpus);
        subN[t] = rankLo + static_cast<int>(static_cast<long long>(rankN) * (t + 1) / numGpus) - subBegin[t];

        CUDA_CHECK(cudaSetDevice(devices[t]));
        CUDA_CHECK(cudaMalloc(&dPos[t], sizeof(Vec3) * static_cast<size_t>(std::max(n, 1))));
        CUDA_CHECK(cudaMalloc(&dVel[t], sizeof(Vec3) * static_cast<size_t>(std::max(subN[t], 1))));
        CUDA_CHECK(cudaMemcpy(dVel[t], vel.data() + subBegin[t], sizeof(Vec3) * subN[t], cudaMemcpyHostToDevice));
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();
    std::chrono::high_resolution_clock::time_point end;

    #pragma omp parallel num_threads(numGpus)
    {
        const int t = omp_get_thread_num();
        CUDA_CHECK(cudaSetDevice(devices[t]));
        const int grid = (subN[t] + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

        for (int step = 0; step < numSteps; ++step) {
            CUDA_CHECK(cudaMemcpy(dPos[t], pos.data(), sizeof(Vec3) * n, cudaMemcpyHostToDevice));
            #pragma omp barrier // all GPUs have read the current positions before any host update

            if (subN[t] > 0) {
                computeForcesKernel<<<grid, THREADS_PER_BLOCK>>>(dPos[t], dVel[t], subBegin[t], subN[t], n);
                integrateKernel<<<grid, THREADS_PER_BLOCK>>>(dPos[t], dVel[t], subBegin[t], subN[t]);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(pos.data() + subBegin[t], dPos[t] + subBegin[t],
                                      sizeof(Vec3) * subN[t], cudaMemcpyDeviceToHost));
            }
            #pragma omp barrier // all sub-blocks of this rank are in host memory

            #pragma omp master
            MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, pos.data(), counts.data(), displs.data(), vec3Type, MPI_COMM_WORLD);
            #pragma omp barrier
        }

        #pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            end = std::chrono::high_resolution_clock::now();
        }

        // Collect final velocities for output and validation
        if (subN[t] > 0) {
            CUDA_CHECK(cudaMemcpy(vel.data() + subBegin[t], dVel[t], sizeof(Vec3) * subN[t], cudaMemcpyDeviceToHost));
        }
        #pragma omp barrier
        #pragma omp master
        MPI_Allgatherv(MPI_IN_PLACE, 0, vec3Type, vel.data(), counts.data(), displs.data(), vec3Type, MPI_COMM_WORLD);
        #pragma omp barrier
    }

    for (int t = 0; t < numGpus; ++t) {
        CUDA_CHECK(cudaSetDevice(devices[t]));
        CUDA_CHECK(cudaFree(dPos[t]));
        CUDA_CHECK(cudaFree(dVel[t]));
    }
    MPI_Type_free(&vec3Type);

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    int rc = 0;
    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(n) * 6);
            for (int i = 0; i < n; ++i) {
                bodyData.push_back(pos[i].x);
                bodyData.push_back(pos[i].y);
                bodyData.push_back(pos[i].z);
                bodyData.push_back(vel[i].x);
                bodyData.push_back(vel[i].y);
                bodyData.push_back(vel[i].z);
            }
            print_results(bodyData, "Bodies");
        }

        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(pos, vel)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(pos, vel);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }
    return rc;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int rc = runBenchmark(argc, argv, rank, size);
    MPI_Bcast(&rc, 1, MPI_INT, 0, MPI_COMM_WORLD);

    MPI_Finalize();
    return rc;
}
