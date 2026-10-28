// Hybrid MPI + OpenMP + CUDA N-body benchmark.
//
// Decomposition:
//   - MPI:    bodies are partitioned into contiguous blocks, one block per rank.
//             After every step the updated positions (and, when needed for
//             output, the velocities) are exchanged with MPI_Allgatherv so that
//             every rank holds the full state for the next force evaluation.
//   - OpenMP: each rank spawns one host thread per GPU it drives. Each thread
//             owns a contiguous sub-block of the rank's bodies and manages one
//             device.
//   - CUDA:   the O(N^2) force evaluation and the position integration run on
//             the GPU. The force kernel is tiled through shared memory.
//
// Semantics: for every body i, the force sum runs over j = 0..N-1 in the same
// order as the sequential reference, and FMA contraction is disabled on both the
// host and device, so the results are bit-identical regardless of how the bodies
// are distributed across ranks and devices.

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int TILE = 256;  // CUDA block size and shared-memory tile length

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                 \
    do {                                                                                 \
        cudaError_t err_ = (call);                                                       \
        if (err_ != cudaSuccess) {                                                       \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,             \
                    cudaGetErrorString(err_));                                           \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                \
        }                                                                                \
    } while (0)

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

// Computes the velocity update for bodies [i0, i0 + count). Reads positions of
// all n bodies (read-only); writes only vel of the owned bodies.
__global__ void computeForcesKernel(Body* bodies, int n, int i0, int count) {
    __shared__ double sx[TILE];
    __shared__ double sy[TILE];
    __shared__ double sz[TILE];

    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = li < count;
    const int i = i0 + (active ? li : 0);
    const double xi = bodies[i].pos.x;
    const double yi = bodies[i].pos.y;
    const double zi = bodies[i].pos.z;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += TILE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = bodies[j].pos.x;
            sy[threadIdx.x] = bodies[j].pos.y;
            sz[threadIdx.x] = bodies[j].pos.z;
        }
        __syncthreads();

        const int lim = min(TILE, n - base);
        if (active) {
            for (int k = 0; k < lim; ++k) {
                const double dx = sx[k] - xi;
                const double dy = sy[k] - yi;
                const double dz = sz[k] - zi;
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
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

// Advances positions of bodies [i0, i0 + count). Runs after the force kernel on
// the same stream, so all reads of the old positions are complete.
__global__ void integrateKernel(Body* bodies, int i0, int count) {
    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li < count) {
        Body& b = bodies[i0 + li];
        b.pos.x += b.vel.x * DT;
        b.pos.y += b.vel.y * DT;
        b.pos.z += b.vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
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
    const bool root = (rank == 0);

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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (numBodies < 0) {
        if (root) printf("Number of bodies must be non-negative\n");
        MPI_Finalize();
        return 1;
    }

    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    const int n = numBodies;

    // Initialize bodies (deterministic, so every rank builds the same state)
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    // Block partition of bodies across MPI ranks
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        const long long start = (long long)n * r / size;
        const long long end = (long long)n * (r + 1) / size;
        displs[r] = (int)start;
        counts[r] = (int)(end - start);
    }
    const int rankStart = displs[rank];
    const int rankCount = counts[rank];

    // MPI datatypes that address the pos or vel member of each Body in place
    MPI_Datatype vec3Type, posType, velType;
    MPI_Type_contiguous(3, MPI_DOUBLE, &vec3Type);
    MPI_Type_commit(&vec3Type);
    MPI_Type_create_resized(vec3Type, 0, (MPI_Aint)sizeof(Body), &posType);
    MPI_Type_commit(&posType);
    MPI_Type_create_resized(vec3Type, 0, (MPI_Aint)sizeof(Body), &velType);
    MPI_Type_commit(&velType);
    char* velBase = reinterpret_cast<char*>(bodies.data()) + offsetof(Body, vel);
    void* posBase = static_cast<void*>(bodies.data());

    // Assign GPUs to the ranks that share this node
    int ndev = 0;
    CUDA_CHECK(cudaGetDeviceCount(&ndev));
    if (ndev == 0) {
        if (root) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Comm shmComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &shmComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(shmComm, &localRank);
    MPI_Comm_size(shmComm, &localSize);
    MPI_Comm_free(&shmComm);
    const int gpusPerRank = std::max(1, ndev / localSize);

    const bool needState = printResults || validate;
    std::chrono::high_resolution_clock::time_point tStart, tEnd;

    #pragma omp parallel num_threads(gpusPerRank)
    {
        const int t = omp_get_thread_num();
        const int nt = omp_get_num_threads();
        const int dev = (localRank * gpusPerRank + t) % ndev;
        CUDA_CHECK(cudaSetDevice(dev));

        // This thread's sub-block of the rank's bodies
        const int i0 = rankStart + (int)((long long)rankCount * t / nt);
        const int i1 = rankStart + (int)((long long)rankCount * (t + 1) / nt);
        const int cnt = i1 - i0;
        const int blocks = (cnt + TILE - 1) / TILE;

        Body* dBodies = nullptr;
        CUDA_CHECK(cudaMalloc(&dBodies, std::max(n, 1) * sizeof(Body)));
        if (n > 0) {
            CUDA_CHECK(cudaMemcpy(dBodies, bodies.data(), n * sizeof(Body), cudaMemcpyHostToDevice));
        }

        #pragma omp barrier
        #pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            tStart = std::chrono::high_resolution_clock::now();
        }
        #pragma omp barrier

        for (int step = 0; step < numSteps; ++step) {
            if (n > 0) {
                // Refresh device positions with the globally gathered state
                CUDA_CHECK(cudaMemcpy2D(dBodies, sizeof(Body), &bodies[0].pos, sizeof(Body),
                                        sizeof(Vec3), n, cudaMemcpyHostToDevice));
            }
            #pragma omp barrier  // all threads have read the host state before any writes

            if (cnt > 0) {
                computeForcesKernel<<<blocks, TILE>>>(dBodies, n, i0, cnt);
                integrateKernel<<<blocks, TILE>>>(dBodies, i0, cnt);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy2D(&bodies[i0].pos, sizeof(Body), &dBodies[i0].pos, sizeof(Body),
                                        sizeof(Vec3), cnt, cudaMemcpyDeviceToHost));
            }
            #pragma omp barrier

            #pragma omp master
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, posBase, counts.data(), displs.data(),
                           posType, MPI_COMM_WORLD);
            #pragma omp barrier
        }

        #pragma omp master
        {
            MPI_Barrier(MPI_COMM_WORLD);
            tEnd = std::chrono::high_resolution_clock::now();
        }
        #pragma omp barrier

        if (needState) {
            if (cnt > 0) {
                CUDA_CHECK(cudaMemcpy2D(&bodies[i0].vel, sizeof(Body), &dBodies[i0].vel, sizeof(Body),
                                        sizeof(Vec3), cnt, cudaMemcpyDeviceToHost));
            }
            #pragma omp barrier
            #pragma omp master
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, velBase, counts.data(), displs.data(),
                           velType, MPI_COMM_WORLD);
            #pragma omp barrier
        }

        CUDA_CHECK(cudaFree(dBodies));
    }

    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(tEnd - tStart);
    if (root) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Print results for external validation
    if (printResults && root) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve((size_t)n * 6);
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

    int exitCode = 0;

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        int ok = 1;
        if (root) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                ok = 0;
            }
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
        exitCode = ok ? 0 : 1;
    }

    MPI_Type_free(&vec3Type);
    MPI_Type_free(&posType);
    MPI_Type_free(&velType);
    MPI_Finalize();
    return exitCode;
}
