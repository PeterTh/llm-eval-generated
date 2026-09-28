// Hybrid MPI + OpenMP + CUDA n-body simulation.
//
// Decomposition:
//  - MPI: bodies are block-partitioned across ranks; each rank owns a
//    contiguous slice and binds to one GPU (selected by node-local rank).
//    Updated positions are exchanged every step with MPI_Allgatherv.
//  - CUDA: the O(n^2) force computation and integration of the local slice
//    run in a fused, shared-memory-tiled kernel. Tiles are processed in
//    ascending j order, preserving the original per-body summation order.
//  - OpenMP: host-side O(n^2) energy computation, validation, and final
//    state assembly/serialization on the root rank.

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 256;

#define CUDA_CHECK(call)                                                              \
    do {                                                                              \
        cudaError_t err_ = (call);                                                    \
        if (err_ != cudaSuccess) {                                                    \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_),     \
                    __FILE__, __LINE__);                                              \
            MPI_Abort(MPI_COMM_WORLD, 1);                                             \
        }                                                                             \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

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

// Fused force computation + integration for the local slice
// [offset, offset + localN) against all n bodies. Reads the position
// snapshot `pos`, writes updated velocities and new positions for the
// local slice only (matching the original computeForces-then-integrate
// step semantics, since forces depend only on the position snapshot).
__global__ void bodyStepKernel(const double3* __restrict__ pos, double3* vel,
                               double3* posOut, int n, int offset, int localN) {
    __shared__ double3 shPos[BLOCK_SIZE];

    const int li = blockIdx.x * blockDim.x + threadIdx.x;
    double3 p = {0.0, 0.0, 0.0};
    if (li < localN) p = pos[offset + li];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        const int j = tile + threadIdx.x;
        if (j < n) shPos[threadIdx.x] = pos[j];
        __syncthreads();

        const int tileSize = min(BLOCK_SIZE, n - tile);
        if (li < localN) {
            #pragma unroll 8
            for (int k = 0; k < tileSize; ++k) {
                const double dx = shPos[k].x - p.x;
                const double dy = shPos[k].y - p.y;
                const double dz = shPos[k].z - p.z;
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

    if (li < localN) {
        double3 v = vel[li];
        v.x += DT * Fx;
        v.y += DT * Fy;
        v.z += DT * Fz;
        vel[li] = v;
        posOut[li] = {p.x + v.x * DT, p.y + v.y * DT, p.z + v.z * DT};
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const long n = static_cast<long>(bodies.size());

    // Kinetic energy (assuming unit mass)
    #pragma omp parallel for reduction(+ : energy)
    for (long i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies)
    #pragma omp parallel for reduction(+ : energy) schedule(dynamic, 64)
    for (long i = 0; i < n; ++i) {
        for (long j = i + 1; j < n; ++j) {
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
    const long n = static_cast<long>(bodies.size());
    int failed = 0;

    #pragma omp parallel for reduction(| : failed)
    for (long i = 0; i < n; ++i) {
        const auto& body = bodies[i];
        // Check for NaN or Inf values
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            failed |= 1;
            continue;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos.x) > maxPos || std::abs(body.pos.y) > maxPos || std::abs(body.pos.z) > maxPos) {
            failed |= 2;
        }
        if (std::abs(body.vel.x) > maxVel || std::abs(body.vel.y) > maxVel || std::abs(body.vel.z) > maxVel) {
            failed |= 4;
        }
    }

    if (failed & 1) printf("Validation failed: found NaN or Inf value in body state\n");
    if (failed & 2) printf("Validation failed: body position exceeds reasonable bounds\n");
    if (failed & 4) printf("Validation failed: body velocity exceeds reasonable bounds\n");
    return failed == 0;
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

    int rank = 0, numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    int parseStatus = 0;  // 0 = run, 1 = help, 2 = error

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
            parseStatus = 1;
            break;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            parseStatus = 2;
            break;
        }
    }
    if (parseStatus != 0) {
        MPI_Finalize();
        return parseStatus == 1 ? 0 : 1;
    }

    // Bind each rank to a GPU based on its node-local rank.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        if (rank == 0) fprintf(stderr, "No CUDA devices available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % deviceCount));

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize bodies (deterministic; every rank generates the same state)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Block partition of bodies across ranks (in units of doubles for MPI).
    std::vector<int> counts(numRanks), displs(numRanks);
    {
        const int base = numBodies / numRanks;
        const int rem = numBodies % numRanks;
        int off = 0;
        for (int r = 0; r < numRanks; ++r) {
            const int c = base + (r < rem ? 1 : 0);
            counts[r] = c * 3;
            displs[r] = off * 3;
            off += c;
        }
    }
    const int localN = counts[rank] / 3;
    const int offset = displs[rank] / 3;

    // Host staging buffer (pinned) holding all positions, plus device buffers.
    double3* h_pos = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_pos, numBodies * sizeof(double3)));
    double3* d_pos = nullptr;
    double3* d_vel = nullptr;
    double3* d_posOut = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pos, numBodies * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&d_vel, (localN > 0 ? localN : 1) * sizeof(double3)));
    CUDA_CHECK(cudaMalloc(&d_posOut, (localN > 0 ? localN : 1) * sizeof(double3)));

    {
        std::vector<double3> localVel(localN > 0 ? localN : 1);
        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            h_pos[i] = {bodies[i].pos.x, bodies[i].pos.y, bodies[i].pos.z};
        }
        for (int i = 0; i < localN; ++i) {
            const auto& v = bodies[offset + i].vel;
            localVel[i] = {v.x, v.y, v.z};
        }
        CUDA_CHECK(cudaMemcpy(d_pos, h_pos, numBodies * sizeof(double3), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vel, localVel.data(), (localN > 0 ? localN : 1) * sizeof(double3),
                              cudaMemcpyHostToDevice));
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const int numBlocks = (localN + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int step = 0; step < numSteps; ++step) {
        if (localN > 0) {
            bodyStepKernel<<<numBlocks, BLOCK_SIZE>>>(d_pos, d_vel, d_posOut,
                                                      numBodies, offset, localN);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpy(h_pos + offset, d_posOut, localN * sizeof(double3),
                                  cudaMemcpyDeviceToHost));
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, h_pos, counts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        CUDA_CHECK(cudaMemcpy(d_pos, h_pos, numBodies * sizeof(double3), cudaMemcpyHostToDevice));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    // Gather final velocities to rank 0 (final positions are already
    // replicated in h_pos by the last allgather).
    std::vector<double3> localVel(localN > 0 ? localN : 1);
    CUDA_CHECK(cudaMemcpy(localVel.data(), d_vel, (localN > 0 ? localN : 1) * sizeof(double3),
                          cudaMemcpyDeviceToHost));
    std::vector<double3> allVel(rank == 0 ? numBodies : 1);
    MPI_Gatherv(localVel.data(), localN * 3, MPI_DOUBLE, allVel.data(), counts.data(),
                displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    int exitCode = 0;
    if (rank == 0) {
        // Assemble the final full body state on the root rank.
        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = {h_pos[i].x, h_pos[i].y, h_pos[i].z};
            bodies[i].vel = {allVel[i].x, allVel[i].y, allVel[i].z};
        }

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData(numBodies * 6);
            #pragma omp parallel for
            for (int i = 0; i < numBodies; ++i) {
                bodyData[i * 6 + 0] = bodies[i].pos.x;
                bodyData[i * 6 + 1] = bodies[i].pos.y;
                bodyData[i * 6 + 2] = bodies[i].pos.z;
                bodyData[i * 6 + 3] = bodies[i].vel.x;
                bodyData[i * 6 + 4] = bodies[i].vel.y;
                bodyData[i * 6 + 5] = bodies[i].vel.z;
            }
            print_results(bodyData, "Bodies");
        }

        // Validation: check that simulation produces finite, reasonable values
        if (validate) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
    }
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    CUDA_CHECK(cudaFree(d_posOut));
    CUDA_CHECK(cudaFreeHost(h_pos));

    MPI_Finalize();
    return exitCode;
}
