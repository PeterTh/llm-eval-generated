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
// Threads per block and bodies per shared-memory tile. Small blocks spread the
// work across all SMs more evenly for the per-GPU body slices.
constexpr int TILE = 64;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Bodies are exchanged over MPI as raw doubles
static_assert(sizeof(Body) == 6 * sizeof(double), "Body must be six packed doubles");

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,              \
                    cudaGetErrorString(err_));                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
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

// Each thread owns one body i in [offset, offset + count) and accumulates the
// force from every j in ascending order, exactly as the serial reference does.
// Positions are staged through shared memory one tile at a time; the tiles are
// visited in order, so the summation order (and therefore the result) is unchanged.
__global__ void computeForcesKernel(Body* bodies, int n, int offset, int count) {
    __shared__ double sx[TILE], sy[TILE], sz[TILE];

    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = k < count;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (active) {
        const Body& self = bodies[offset + k];
        xi = self.pos.x;
        yi = self.pos.y;
        zi = self.pos.z;
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int j0 = 0; j0 < n; j0 += TILE) {
        const int jn = (n - j0 < TILE) ? (n - j0) : TILE;

        if ((int)threadIdx.x < jn) {
            const Body& b = bodies[j0 + threadIdx.x];
            sx[threadIdx.x] = b.pos.x;
            sy[threadIdx.x] = b.pos.y;
            sz[threadIdx.x] = b.pos.z;
        }
        __syncthreads();

        if (active) {
            for (int t = 0; t < jn; ++t) {
                const double dx = sx[t] - xi;
                const double dy = sy[t] - yi;
                const double dz = sz[t] - zi;
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
        Body& self = bodies[offset + k];
        self.vel.x += DT * Fx;
        self.vel.y += DT * Fy;
        self.vel.z += DT * Fz;
    }
}

__global__ void integrateKernel(Body* bodies, int offset, int count) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) {
        return;
    }
    Body& b = bodies[offset + k];
    b.pos.x += b.vel.x * DT;
    b.pos.y += b.vel.y * DT;
    b.pos.z += b.vel.z * DT;
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

// Runs on every MPI rank. Returns the process exit code.
int runNBody(int argc, char** argv, int rank, int size) {
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
            if (rank == 0) {
                printUsage(argv[0]);
            }
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (numBodies < 0) {
        if (rank == 0) {
            printf("Number of bodies must be non-negative\n");
        }
        return 1;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Assign GPUs: ranks sharing a node split its devices, and each rank drives
    // one OpenMP thread (and one GPU) per device it owns.
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_size(nodeComm, &localSize);
    MPI_Comm_free(&nodeComm);

    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices found on rank %d\n", rank);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    omp_set_dynamic(0);
    const int threadsPerRank = std::max(1, std::min(omp_get_max_threads(), numDevices / localSize));

    // Global worker numbering: rank r owns workers [firstWorker, firstWorker + threadsPerRank)
    std::vector<int> threadsOf(size);
    MPI_Allgather(&threadsPerRank, 1, MPI_INT, threadsOf.data(), 1, MPI_INT, MPI_COMM_WORLD);
    int totalWorkers = 0, firstWorker = 0;
    for (int r = 0; r < size; ++r) {
        if (r < rank) {
            firstWorker += threadsOf[r];
        }
        totalWorkers += threadsOf[r];
    }

    // Body range owned by global worker w is [bodyBegin(w), bodyBegin(w + 1))
    auto bodyBegin = [&](int w) {
        return static_cast<int>((static_cast<long long>(w) * numBodies) / totalWorkers);
    };

    // Per-rank segments of the body array, as MPI_Allgatherv counts/displacements (in doubles)
    std::vector<int> counts(size), displs(size);
    {
        int fw = 0;
        for (int r = 0; r < size; ++r) {
            const int begin = bodyBegin(fw);
            const int end = bodyBegin(fw + threadsOf[r]);
            counts[r] = 6 * (end - begin);
            displs[r] = 6 * begin;
            fw += threadsOf[r];
        }
    }

    // Every rank holds the full state. Initialization is deterministic, so no broadcast is needed.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // One device buffer and body slice per local worker
    struct Worker {
        int device;
        int begin;
        int count;
        Body* dBodies;
    };
    std::vector<Worker> workers(threadsPerRank);
    for (int t = 0; t < threadsPerRank; ++t) {
        const int g = firstWorker + t;
        workers[t].device = (localRank * threadsPerRank + t) % numDevices;
        workers[t].begin = bodyBegin(g);
        workers[t].count = bodyBegin(g + 1) - workers[t].begin;
        workers[t].dBodies = nullptr;
    }

    // Allocate device memory before timing starts
    #pragma omp parallel num_threads(threadsPerRank)
    {
        Worker& w = workers[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(w.device));
        CUDA_CHECK(cudaMalloc(&w.dBodies, sizeof(Body) * std::max(1, numBodies)));
    }

    // Run simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel num_threads(threadsPerRank)
    {
        Worker& w = workers[omp_get_thread_num()];
        CUDA_CHECK(cudaSetDevice(w.device));

        const dim3 block(TILE);
        const dim3 grid((w.count + TILE - 1) / TILE);

        for (int step = 0; step < numSteps; ++step) {
            // Upload the full state. The barrier keeps every worker's upload (a read of
            // `bodies`) ahead of the slice downloads (writes to `bodies`) below.
            CUDA_CHECK(cudaMemcpy(w.dBodies, bodies.data(), sizeof(Body) * numBodies, cudaMemcpyHostToDevice));
            #pragma omp barrier

            if (w.count > 0) {
                computeForcesKernel<<<grid, block>>>(w.dBodies, numBodies, w.begin, w.count);
                integrateKernel<<<grid, block>>>(w.dBodies, w.begin, w.count);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaMemcpy(bodies.data() + w.begin, w.dBodies + w.begin,
                                      sizeof(Body) * w.count, cudaMemcpyDeviceToHost));
            }
            #pragma omp barrier

            // Exchange updated slices between ranks; each rank's segment is already in place
            #pragma omp master
            {
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, bodies.data(), counts.data(),
                               displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
            }
            #pragma omp barrier
        }

        CUDA_CHECK(cudaFree(w.dBodies));
    }

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", duration.count());
    }

    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
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

    // Validation runs on rank 0, whose state matches every other rank's; the verdict is broadcast
    int exitCode = 0;
    if (validate) {
        int valid = 0;
        if (rank == 0) {
            printf("Validating simulation results...\n");

            if (validateSimulation(bodies)) {
                // Report final energy for reference
                double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                valid = 1;
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        exitCode = valid ? 0 : 1;
    }

    return exitCode;
}

int main(int argc, char** argv) {
    // Only the OpenMP master thread calls MPI, so FUNNELED is sufficient
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    if (provided < MPI_THREAD_FUNNELED) {
        fprintf(stderr, "MPI implementation does not provide MPI_THREAD_FUNNELED\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    const int rc = runNBody(argc, argv, rank, size);

    MPI_Finalize();
    return rc;
}
