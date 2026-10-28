// Hybrid MPI + OpenMP + CUDA N-body benchmark.
//
// Decomposition:
//   - MPI: bodies are partitioned into contiguous blocks, one block per rank.
//     Each step the position blocks are exchanged with MPI_Allgatherv so that
//     every rank holds the full set of positions for the force computation.
//   - OpenMP: each rank runs one host thread per GPU it drives. Each thread owns
//     one sub-block of the rank's bodies and controls its own CUDA device.
//     OpenMP is also used for the O(N^2) energy evaluation on rank 0.
//   - CUDA: the O(N^2) force evaluation is a tiled kernel. Every body's force sum
//     runs over j in ascending order, exactly as in the sequential reference, so
//     the arithmetic order per body is unchanged.

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
constexpr int FORCE_BLOCK = 256;

struct Vec3 {
    double x, y, z;
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,          \
                    cudaGetErrorString(err_));                                        \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

// Splits [0, total) into `parts` contiguous blocks; returns block `idx`.
static void partitionRange(long long total, int parts, int idx, int& offset, int& count) {
    const long long base = total / parts;
    const long long rem = total % parts;
    offset = static_cast<int>(idx * base + std::min<long long>(idx, rem));
    count = static_cast<int>(base + (idx < rem ? 1 : 0));
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

// Force + velocity update for bodies [first, first + count).
// `pos` holds all n positions (as of the start of the step); `vel` holds the
// velocities of the `count` locally owned bodies.
__global__ void computeForcesKernel(const Vec3* __restrict__ pos, Vec3* __restrict__ vel,
                                    int n, int first, int count) {
    __shared__ Vec3 tile[FORCE_BLOCK];

    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = li < count;
    const Vec3 pi = active ? pos[first + li] : Vec3{0.0, 0.0, 0.0};

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += FORCE_BLOCK) {
        const int j = base + threadIdx.x;
        if (j < n) {
            tile[threadIdx.x] = pos[j];
        }
        __syncthreads();

        const int limit = min(FORCE_BLOCK, n - base);
#pragma unroll 8
        for (int k = 0; k < limit; ++k) {
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

// pos += vel * DT for `count` bodies.
__global__ void integrateKernel(Vec3* __restrict__ pos, const Vec3* __restrict__ vel, int count) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < count) {
        pos[i].x += vel[i].x * DT;
        pos[i].y += vel[i].y * DT;
        pos[i].z += vel[i].z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const long n = static_cast<long>(bodies.size());

    // Kinetic energy (assuming unit mass)
    double energy = 0.0;
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                         body.vel.y * body.vel.y +
                         body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies), parallelized over rows.
    double potential = 0.0;
#pragma omp parallel for schedule(dynamic, 64) reduction(+ : potential)
    for (long i = 0; i < n; ++i) {
        for (long j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential -= 1.0 / dist;
        }
    }

    return energy + potential;
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

int main(int argc, char** argv) {
    int provided = MPI_THREAD_SINGLE;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0, nranks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &nranks);

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

    if (numBodies < 0) {
        if (rank == 0) printf("Number of bodies must be non-negative\n");
        MPI_Finalize();
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Node-local topology: one OpenMP thread (and one GPU) per device this rank drives.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices <= 0) {
        fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    const int threadsPerRank = std::max(1, numDevices / localSize);
    omp_set_dynamic(0);
    omp_set_num_threads(threadsPerRank);

    // Body partition across MPI ranks (counts/displacements in units of doubles).
    std::vector<int> posCounts(nranks), posDispls(nranks);
    int myOffset = 0, myCount = 0;
    for (int r = 0; r < nranks; ++r) {
        int off = 0, cnt = 0;
        partitionRange(numBodies, nranks, r, off, cnt);
        posCounts[r] = 3 * cnt;
        posDispls[r] = 3 * off;
        if (r == rank) {
            myOffset = off;
            myCount = cnt;
        }
    }

    // Initialize bodies (identical on every rank, so no initial scatter is needed).
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // posHost: complete positions, read by every device upload; written only by the master thread.
    // posStage: each thread writes its own slice here after the kernels, so posHost stays read-only
    //           while other threads may still be uploading it.
    // velHost: each thread writes only its own slice (disjoint), gathered in place at the end.
    std::vector<Vec3> posHost(numBodies), posStage(numBodies), velHost(numBodies);
    for (int i = 0; i < numBodies; ++i) {
        posHost[i] = bodies[i].pos;
        velHost[i] = bodies[i].vel;
    }
    double* velDouble = reinterpret_cast<double*>(velHost.data());

    double localSeconds = 0.0;

#pragma omp parallel num_threads(threadsPerRank)
    {
        if (omp_get_num_threads() != threadsPerRank) {
            fprintf(stderr, "Failed to create %d OpenMP threads\n", threadsPerRank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        const int t = omp_get_thread_num();
        const int device = (localRank * threadsPerRank + t) % numDevices;
        CUDA_CHECK(cudaSetDevice(device));

        int subOffset = 0, subCount = 0;
        partitionRange(myCount, threadsPerRank, t, subOffset, subCount);
        const int first = myOffset + subOffset;
        const int cnt = subCount;

        // Device state: full position array (for force evaluation) and velocities of owned bodies.
        Vec3* dPos = nullptr;
        Vec3* dVel = nullptr;
        CUDA_CHECK(cudaMalloc(&dPos, sizeof(Vec3) * static_cast<size_t>(std::max(numBodies, 1))));
        if (cnt > 0) {
            CUDA_CHECK(cudaMalloc(&dVel, sizeof(Vec3) * static_cast<size_t>(cnt)));
            CUDA_CHECK(cudaMemcpy(dVel, velHost.data() + first, sizeof(Vec3) * cnt, cudaMemcpyHostToDevice));
        }

#pragma omp barrier
#pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#pragma omp barrier

        std::chrono::high_resolution_clock::time_point start;
#pragma omp master
        {
            start = std::chrono::high_resolution_clock::now();
        }
#pragma omp barrier

        const int blocks = (cnt + FORCE_BLOCK - 1) / FORCE_BLOCK;

        for (int step = 0; step < numSteps; ++step) {
            if (numBodies > 0) {
                CUDA_CHECK(cudaMemcpy(dPos, posHost.data(), sizeof(Vec3) * numBodies, cudaMemcpyHostToDevice));
            }
            if (cnt > 0) {
                computeForcesKernel<<<blocks, FORCE_BLOCK>>>(dPos, dVel, numBodies, first, cnt);
                integrateKernel<<<blocks, FORCE_BLOCK>>>(dPos + first, dVel, cnt);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(posStage.data() + first, dPos + first, sizeof(Vec3) * cnt,
                                      cudaMemcpyDeviceToHost));
            }
#pragma omp barrier
#pragma omp master
            {
                // Gather the rank's block (staged above) into the full host position array.
                MPI_Allgatherv(reinterpret_cast<double*>(posStage.data()) + posDispls[rank], posCounts[rank],
                               MPI_DOUBLE, reinterpret_cast<double*>(posHost.data()), posCounts.data(),
                               posDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            }
#pragma omp barrier
        }

#pragma omp master
        {
            auto end = std::chrono::high_resolution_clock::now();
            localSeconds = std::chrono::duration<double>(end - start).count();
        }
#pragma omp barrier

        // Gather final velocities so every rank holds the complete state.
        if (cnt > 0) {
            CUDA_CHECK(cudaMemcpy(velHost.data() + first, dVel, sizeof(Vec3) * cnt, cudaMemcpyDeviceToHost));
        }
#pragma omp barrier
#pragma omp master
        {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, velDouble, posCounts.data(),
                           posDispls.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }
#pragma omp barrier

        if (dVel) CUDA_CHECK(cudaFree(dVel));
        CUDA_CHECK(cudaFree(dPos));
    }

    double maxSeconds = 0.0;
    MPI_Reduce(&localSeconds, &maxSeconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::duration<double>(maxSeconds));

    for (int i = 0; i < numBodies; ++i) {
        bodies[i].pos = posHost[i];
        bodies[i].vel = velHost[i];
    }

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(static_cast<size_t>(numBodies) * 6);
            for (const auto& body : bodies) {
                bodyData.push_back(body.pos.x);
                bodyData.push_back(body.pos.y);
                bodyData.push_back(body.pos.z);
                bodyData.push_back(body.vel.x);
                bodyData.push_back(body.vel.y);
                bodyData.push_back(body.vel.z);
            }
            print_results(bodyData, "Bodies");
        }
    }

    int exitCode = 0;
    // Validation: check that simulation produces finite, reasonable values
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference (host-side, uses this node's spare cores)
            omp_set_num_threads(std::max(1, omp_get_num_procs() / localSize));
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
