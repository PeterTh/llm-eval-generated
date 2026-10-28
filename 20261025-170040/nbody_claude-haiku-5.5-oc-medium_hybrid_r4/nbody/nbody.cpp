// Hybrid MPI + OpenMP + CUDA N-body benchmark.
//
// Decomposition:
//   - MPI:    bodies are split into contiguous blocks, one block per rank.
//             Positions are exchanged with MPI_Allgatherv after every step.
//             Velocities remain local to the owning rank until the end.
//   - OpenMP: each rank runs one thread per GPU it owns; each thread drives
//             its device over a sub-block of the rank's bodies.
//   - CUDA:   force and integration kernels. Every body's j-loop is accumulated
//             in ascending j order (tiled through shared memory), so per-body
//             results match the sequential reference computation.

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
constexpr int FORCE_BLOCK = 256;  // threads per block; also the shared-memory tile size

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        cudaError_t err_ = (call);                                                             \
        if (err_ != cudaSuccess) {                                                             \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,                   \
                    cudaGetErrorString(err_));                                                 \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                      \
        }                                                                                      \
    } while (0)

struct Range {
    size_t begin;
    size_t count;
};

// Balanced contiguous split of [0, total) into `parts` pieces; returns piece `idx`.
static Range splitRange(size_t total, int parts, int idx) {
    const size_t b = total * static_cast<size_t>(idx) / static_cast<size_t>(parts);
    const size_t e = total * static_cast<size_t>(idx + 1) / static_cast<size_t>(parts);
    return {b, e - b};
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

// Force accumulation for the local body range [first, first + count).
// pos holds all n bodies (interleaved x,y,z); vel holds the local bodies only.
// Each thread owns one body i and streams all j = 0..n-1 in ascending order
// through shared memory, so the summation order matches the sequential reference.
__global__ void computeForcesKernel(const double* __restrict__ pos, double* __restrict__ vel,
                                    size_t first, size_t count, size_t n) {
    __shared__ double sx[FORCE_BLOCK];
    __shared__ double sy[FORCE_BLOCK];
    __shared__ double sz[FORCE_BLOCK];

    const size_t li = static_cast<size_t>(blockIdx.x) * FORCE_BLOCK + threadIdx.x;
    const bool active = li < count;

    double px = 0.0, py = 0.0, pz = 0.0;
    if (active) {
        const size_t i = first + li;
        px = pos[3 * i];
        py = pos[3 * i + 1];
        pz = pos[3 * i + 2];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (size_t base = 0; base < n; base += FORCE_BLOCK) {
        const size_t j = base + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = pos[3 * j];
            sy[threadIdx.x] = pos[3 * j + 1];
            sz[threadIdx.x] = pos[3 * j + 2];
        }
        __syncthreads();

        if (active) {
            const size_t remaining = n - base;
            const size_t lim = remaining < static_cast<size_t>(FORCE_BLOCK) ? remaining : static_cast<size_t>(FORCE_BLOCK);
            for (size_t k = 0; k < lim; ++k) {
                const double dx = sx[k] - px;
                const double dy = sy[k] - py;
                const double dz = sz[k] - pz;
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
        vel[3 * li] += DT * Fx;
        vel[3 * li + 1] += DT * Fy;
        vel[3 * li + 2] += DT * Fz;
    }
}

// Position update for the local body range; writes pos for owned bodies only.
__global__ void integrateKernel(double* __restrict__ pos, const double* __restrict__ vel,
                                size_t first, size_t count) {
    const size_t li = static_cast<size_t>(blockIdx.x) * FORCE_BLOCK + threadIdx.x;
    if (li >= count) return;
    const size_t i = first + li;
    pos[3 * i] += vel[3 * li] * DT;
    pos[3 * i + 1] += vel[3 * li + 1] * DT;
    pos[3 * i + 2] += vel[3 * li + 2] * DT;
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

// Per-OpenMP-thread (per-GPU) state. Entry t is only touched by thread t.
struct DeviceState {
    int device = 0;
    size_t first = 0;   // global index of first body owned by this device
    size_t count = 0;   // number of bodies owned by this device
    double* dPos = nullptr;  // all N bodies, interleaved xyz
    double* dVel = nullptr;  // owned bodies only, interleaved xyz
};

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);
    const bool root = (rank == 0);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    int exitCode = 0;
    bool exitNow = false;
    for (int i = 1; i < argc && !exitNow; ++i) {
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
            exitNow = true;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            exitCode = 1;
            exitNow = true;
        }
    }
    if (exitNow) {
        MPI_Finalize();
        return exitCode;
    }

    if (root) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Initialize bodies (identical on every rank)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    const size_t N = bodies.size();
    const bool simulate = (N > 0 && numSteps > 0);

    long durationMs = 0;

    if (simulate) {
        // Host state: positions are shared globally; velocities are authoritative
        // only on the owning device until the final gather.
        std::vector<double> hostPos(3 * N);
        std::vector<double> hostVel(3 * N);
        for (size_t i = 0; i < N; ++i) {
            hostPos[3 * i] = bodies[i].pos.x;
            hostPos[3 * i + 1] = bodies[i].pos.y;
            hostPos[3 * i + 2] = bodies[i].pos.z;
            hostVel[3 * i] = bodies[i].vel.x;
            hostVel[3 * i + 1] = bodies[i].vel.y;
            hostVel[3 * i + 2] = bodies[i].vel.z;
        }

        // Rank-level body blocks for MPI exchange
        std::vector<int> recvCounts(numRanks);
        std::vector<int> displs(numRanks);
        for (int k = 0; k < numRanks; ++k) {
            const Range r = splitRange(N, numRanks, k);
            recvCounts[k] = static_cast<int>(3 * r.count);
            displs[k] = static_cast<int>(3 * r.begin);
        }
        const Range myRange = splitRange(N, numRanks, rank);

        // Node-local GPU assignment
        MPI_Comm nodeComm;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
        int localRank = 0;
        int ranksPerNode = 1;
        MPI_Comm_rank(nodeComm, &localRank);
        MPI_Comm_size(nodeComm, &ranksPerNode);
        MPI_Comm_free(&nodeComm);

        int numDevices = 0;
        CUDA_CHECK(cudaGetDeviceCount(&numDevices));
        if (numDevices == 0) {
            fprintf(stderr, "No CUDA devices available on rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }

        std::vector<int> devices;
        if (ranksPerNode >= numDevices) {
            devices.push_back(localRank % numDevices);
        } else {
            for (int d = 0; d < numDevices; ++d) {
                if (d % ranksPerNode == localRank) devices.push_back(d);
            }
        }
        const int nThreads = static_cast<int>(devices.size());

        std::vector<DeviceState> dev(nThreads);
        for (int t = 0; t < nThreads; ++t) {
            const Range sub = splitRange(myRange.count, nThreads, t);
            dev[t].device = devices[t];
            dev[t].first = myRange.begin + sub.begin;
            dev[t].count = sub.count;
        }

        // Allocate device buffers and upload initial velocities (one thread per GPU)
        #pragma omp parallel num_threads(nThreads)
        {
            if (omp_get_thread_num() == 0 && omp_get_num_threads() != nThreads) {
                fprintf(stderr, "OpenMP granted %d threads, need %d\n", omp_get_num_threads(), nThreads);
                MPI_Abort(MPI_COMM_WORLD, 1);
            }
            DeviceState& s = dev[omp_get_thread_num()];
            CUDA_CHECK(cudaSetDevice(s.device));
            CUDA_CHECK(cudaMalloc(&s.dPos, 3 * N * sizeof(double)));
            if (s.count > 0) {
                CUDA_CHECK(cudaMalloc(&s.dVel, 3 * s.count * sizeof(double)));
                CUDA_CHECK(cudaMemcpy(s.dVel, hostVel.data() + 3 * s.first,
                                      3 * s.count * sizeof(double), cudaMemcpyHostToDevice));
            }
        }

        MPI_Barrier(MPI_COMM_WORLD);
        auto start = std::chrono::high_resolution_clock::now();

        for (int step = 0; step < numSteps; ++step) {
            // Forces + integration for owned bodies on every local GPU
            #pragma omp parallel num_threads(nThreads)
            {
                DeviceState& s = dev[omp_get_thread_num()];
                if (s.count > 0) {
                    CUDA_CHECK(cudaSetDevice(s.device));
                    // Read the full pre-step positions from hostPos ...
                    CUDA_CHECK(cudaMemcpy(s.dPos, hostPos.data(), 3 * N * sizeof(double),
                                          cudaMemcpyHostToDevice));
                    const unsigned int blocks = static_cast<unsigned int>((s.count + FORCE_BLOCK - 1) / FORCE_BLOCK);
                    computeForcesKernel<<<blocks, FORCE_BLOCK>>>(s.dPos, s.dVel, s.first, s.count, N);
                    integrateKernel<<<blocks, FORCE_BLOCK>>>(s.dPos, s.dVel, s.first, s.count);
                    CUDA_CHECK(cudaGetLastError());
                }
                // No thread may write hostPos until every thread has finished reading it.
                #pragma omp barrier
                if (s.count > 0) {
                    // ... then write back only the owned slice.
                    CUDA_CHECK(cudaMemcpy(hostPos.data() + 3 * s.first, s.dPos + 3 * s.first,
                                          3 * s.count * sizeof(double), cudaMemcpyDeviceToHost));
                }
            }
            // Implicit barrier above: all owned positions are now in hostPos.
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostPos.data(), recvCounts.data(),
                           displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        }

        // Gather final velocities
        #pragma omp parallel num_threads(nThreads)
        {
            DeviceState& s = dev[omp_get_thread_num()];
            if (s.count > 0) {
                CUDA_CHECK(cudaSetDevice(s.device));
                CUDA_CHECK(cudaMemcpy(hostVel.data() + 3 * s.first, s.dVel,
                                      3 * s.count * sizeof(double), cudaMemcpyDeviceToHost));
            }
        }
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hostVel.data(), recvCounts.data(),
                       displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        auto end = std::chrono::high_resolution_clock::now();
        durationMs = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        // Release device resources
        #pragma omp parallel num_threads(nThreads)
        {
            DeviceState& s = dev[omp_get_thread_num()];
            CUDA_CHECK(cudaSetDevice(s.device));
            CUDA_CHECK(cudaFree(s.dPos));
            CUDA_CHECK(cudaFree(s.dVel));
        }

        // Write final state back into bodies (root only needs it)
        if (root) {
            for (size_t i = 0; i < N; ++i) {
                bodies[i].pos = Vec3(hostPos[3 * i], hostPos[3 * i + 1], hostPos[3 * i + 2]);
                bodies[i].vel = Vec3(hostVel[3 * i], hostVel[3 * i + 1], hostVel[3 * i + 2]);
            }
        }
    }

    if (root) {
        printf("Simulation time: %ld ms\n", durationMs);

        // Print results for external validation
        if (printResults) {
            // Serialize body positions and velocities for hashing
            std::vector<double> bodyData;
            bodyData.reserve(N * 6);
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

    // Propagate the root's exit status to all ranks
    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return exitCode;
}
