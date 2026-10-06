#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

// ---------------------------------------------------------------------------
// Error checking helpers
// ---------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        cudaError_t err_ = (call);                                                        \
        if (err_ != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, \
                    __LINE__);                                                            \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                 \
        }                                                                                 \
    } while (0)

constexpr int BLOCK_SIZE = 128;

// ---------------------------------------------------------------------------
// Device kernels
// ---------------------------------------------------------------------------

// Partial force accumulation. Thread (i, c) sums contributions of bodies j in
// chunk c (in increasing j order) acting on local body lo + i. With a single
// chunk this is exactly the original sequential j loop.
__global__ void forceKernel(const double* __restrict__ pos, double* __restrict__ partial, int n, int lo,
                            int nloc, int chunkLen) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int c = blockIdx.y;
    const int jBeg = c * chunkLen;
    const int jEnd = min(n, jBeg + chunkLen);

    double px = 0.0, py = 0.0, pz = 0.0;
    if (i < nloc) {
        const int gi = lo + i;
        px = pos[3 * gi + 0];
        py = pos[3 * gi + 1];
        pz = pos[3 * gi + 2];
    }

    __shared__ double sx[BLOCK_SIZE], sy[BLOCK_SIZE], sz[BLOCK_SIZE];
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = jBeg; tile < jEnd; tile += BLOCK_SIZE) {
        const int j = tile + threadIdx.x;
        if (j < jEnd) {
            sx[threadIdx.x] = pos[3 * j + 0];
            sy[threadIdx.x] = pos[3 * j + 1];
            sz[threadIdx.x] = pos[3 * j + 2];
        }
        __syncthreads();
        const int cnt = min(BLOCK_SIZE, jEnd - tile);
        if (cnt == BLOCK_SIZE) {
#pragma unroll 8
            for (int k = 0; k < BLOCK_SIZE; ++k) {
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
        } else {
            for (int k = 0; k < cnt; ++k) {
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

    if (i < nloc) {
        const size_t base = 3 * ((size_t)c * nloc + i);
        partial[base + 0] = Fx;
        partial[base + 1] = Fy;
        partial[base + 2] = Fz;
    }
}

// Combine partial forces (in chunk order), kick velocities and drift positions
// of the local bodies. Updated local positions are written to posOut.
__global__ void updateKernel(const double* __restrict__ pos, const double* __restrict__ partial,
                             double* __restrict__ vel, double* __restrict__ posOut, int lo, int nloc,
                             int numChunks) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nloc) return;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int c = 0; c < numChunks; ++c) {
        const size_t base = 3 * ((size_t)c * nloc + i);
        Fx += partial[base + 0];
        Fy += partial[base + 1];
        Fz += partial[base + 2];
    }
    double vx = vel[3 * i + 0] + DT * Fx;
    double vy = vel[3 * i + 1] + DT * Fy;
    double vz = vel[3 * i + 2] + DT * Fz;
    vel[3 * i + 0] = vx;
    vel[3 * i + 1] = vy;
    vel[3 * i + 2] = vz;
    const int gi = lo + i;
    posOut[3 * i + 0] = pos[3 * gi + 0] + vx * DT;
    posOut[3 * i + 1] = pos[3 * gi + 1] + vy * DT;
    posOut[3 * i + 2] = pos[3 * gi + 2] + vz * DT;
}

// Potential energy contribution of rows i = rank, rank + size, ... (cyclic
// distribution balances the triangular workload across ranks).
__global__ void potentialKernel(const double* __restrict__ pos, double* __restrict__ rowEnergy, int n,
                                int rank, int size, int numRows) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= numRows) return;
    const int i = rank + r * size;
    const double px = pos[3 * i + 0], py = pos[3 * i + 1], pz = pos[3 * i + 2];
    double e = 0.0;
    for (int j = i + 1; j < n; ++j) {
        const double dx = pos[3 * j + 0] - px;
        const double dy = pos[3 * j + 1] - py;
        const double dz = pos[3 * j + 2] - pz;
        const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        e -= 1.0 / dist;
    }
    rowEnergy[r] = e;
}

// ---------------------------------------------------------------------------
// Distributed simulation state
// ---------------------------------------------------------------------------
struct Simulation {
    int rank = 0, size = 1;
    int n = 0, lo = 0, nloc = 0;
    int numChunks = 1, chunkLen = 1;
    std::vector<int> counts3, displs3;  // per-rank element counts / offsets (3 doubles per body)
    double* hPos = nullptr;             // pinned host copy of all positions (3n)
    double* dPos = nullptr;             // device: all positions (3n)
    double* dPosLoc = nullptr;          // device: updated local positions (3 nloc)
    double* dVel = nullptr;             // device: local velocities (3 nloc)
    double* dPartial = nullptr;         // device: partial forces (3 nloc numChunks)
    cudaStream_t stream{};

    void init(const std::vector<Body>& bodies, int rank_, int size_) {
        rank = rank_;
        size = size_;
        n = (int)bodies.size();
        counts3.resize(size);
        displs3.resize(size);
        for (int r = 0; r < size; ++r) {
            const int rlo = (int)((long long)n * r / size);
            const int rhi = (int)((long long)n * (r + 1) / size);
            counts3[r] = 3 * (rhi - rlo);
            displs3[r] = 3 * rlo;
        }
        lo = displs3[rank] / 3;
        nloc = counts3[rank] / 3;

        CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

        // Split the j range into chunks only when the local body count is too
        // small to fill the device with one thread per body.
        int dev = 0, numSMs = 1;
        CUDA_CHECK(cudaGetDevice(&dev));
        CUDA_CHECK(cudaDeviceGetAttribute(&numSMs, cudaDevAttrMultiProcessorCount, dev));
        const long long targetThreads = (long long)numSMs * 2048;
        if (nloc > 0 && nloc < targetThreads) {
            numChunks = (int)((targetThreads + nloc - 1) / nloc);
            const int maxChunks = std::max(1, n / (4 * BLOCK_SIZE));
            numChunks = std::max(1, std::min(numChunks, maxChunks));
        }
        chunkLen = (n + numChunks - 1) / std::max(1, numChunks);
        numChunks = chunkLen > 0 ? (n + chunkLen - 1) / chunkLen : 1;

        const size_t nAll = std::max<size_t>(1, 3 * (size_t)n);
        const size_t nLoc = std::max<size_t>(1, 3 * (size_t)nloc);
        CUDA_CHECK(cudaMallocHost(&hPos, nAll * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dPos, nAll * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dPosLoc, nLoc * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dVel, nLoc * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&dPartial, nLoc * numChunks * sizeof(double)));

        std::vector<double> hVel(nLoc);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < n; ++i) {
            hPos[3 * i + 0] = bodies[i].pos.x;
            hPos[3 * i + 1] = bodies[i].pos.y;
            hPos[3 * i + 2] = bodies[i].pos.z;
        }
#pragma omp parallel for schedule(static)
        for (int i = 0; i < nloc; ++i) {
            hVel[3 * i + 0] = bodies[lo + i].vel.x;
            hVel[3 * i + 1] = bodies[lo + i].vel.y;
            hVel[3 * i + 2] = bodies[lo + i].vel.z;
        }
        CUDA_CHECK(cudaMemcpy(dPos, hPos, 3 * (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dVel, hVel.data(), 3 * (size_t)nloc * sizeof(double), cudaMemcpyHostToDevice));
    }

    void step() {
        if (nloc > 0) {
            dim3 grid((nloc + BLOCK_SIZE - 1) / BLOCK_SIZE, numChunks);
            forceKernel<<<grid, BLOCK_SIZE, 0, stream>>>(dPos, dPartial, n, lo, nloc, chunkLen);
            updateKernel<<<(nloc + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(
                dPos, dPartial, dVel, dPosLoc, lo, nloc, numChunks);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(hPos + 3 * (size_t)lo, dPosLoc, 3 * (size_t)nloc * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }
        // Exchange the updated positions of all ranks
        if (size > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPos, counts3.data(), displs3.data(), MPI_DOUBLE,
                           MPI_COMM_WORLD);
        }
        if (n > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dPos, hPos, 3 * (size_t)n * sizeof(double), cudaMemcpyHostToDevice, stream));
        }
    }

    // Collect final state into bodies on rank 0
    void gather(std::vector<Body>& bodies) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        std::vector<double> locVel(std::max<size_t>(1, 3 * (size_t)nloc));
        if (nloc > 0) {
            CUDA_CHECK(cudaMemcpy(locVel.data(), dVel, 3 * (size_t)nloc * sizeof(double), cudaMemcpyDeviceToHost));
        }
        std::vector<double> allVel(rank == 0 ? std::max<size_t>(1, 3 * (size_t)n) : 1);
        MPI_Gatherv(locVel.data(), 3 * nloc, MPI_DOUBLE, allVel.data(), counts3.data(), displs3.data(), MPI_DOUBLE, 0,
                    MPI_COMM_WORLD);
        if (rank == 0) {
#pragma omp parallel for schedule(static)
            for (int i = 0; i < n; ++i) {
                bodies[i].pos = Vec3(hPos[3 * i + 0], hPos[3 * i + 1], hPos[3 * i + 2]);
                bodies[i].vel = Vec3(allVel[3 * i + 0], allVel[3 * i + 1], allVel[3 * i + 2]);
            }
        }
    }

    // Total energy: kinetic part from local velocities (OpenMP), potential
    // part from cyclically distributed rows (GPU); reduced across ranks.
    double totalEnergy() {
        std::vector<double> locVel(std::max<size_t>(1, 3 * (size_t)nloc));
        if (nloc > 0) {
            CUDA_CHECK(cudaMemcpy(locVel.data(), dVel, 3 * (size_t)nloc * sizeof(double), cudaMemcpyDeviceToHost));
        }
        double energy = 0.0;
#pragma omp parallel for reduction(+ : energy) schedule(static)
        for (int i = 0; i < nloc; ++i) {
            energy += 0.5 * (locVel[3 * i] * locVel[3 * i] + locVel[3 * i + 1] * locVel[3 * i + 1] +
                             locVel[3 * i + 2] * locVel[3 * i + 2]);
        }

        const int numRows = rank < n ? (n - rank + size - 1) / size : 0;
        if (numRows > 0) {
            double* dRow = nullptr;
            CUDA_CHECK(cudaMalloc(&dRow, numRows * sizeof(double)));
            potentialKernel<<<(numRows + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE, 0, stream>>>(dPos, dRow, n, rank,
                                                                                                size, numRows);
            CUDA_CHECK(cudaGetLastError());
            std::vector<double> rowEnergy(numRows);
            CUDA_CHECK(cudaMemcpyAsync(rowEnergy.data(), dRow, numRows * sizeof(double), cudaMemcpyDeviceToHost,
                                       stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            CUDA_CHECK(cudaFree(dRow));
#pragma omp parallel for reduction(+ : energy) schedule(static)
            for (int r = 0; r < numRows; ++r) energy += rowEnergy[r];
        }
        double total = 0.0;
        MPI_Allreduce(&energy, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
        return total;
    }

    void destroy() {
        cudaFree(dPartial);
        cudaFree(dVel);
        cudaFree(dPosLoc);
        cudaFree(dPos);
        cudaFreeHost(hPos);
        cudaStreamDestroy(stream);
    }
};

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
    MPI_Init(&argc, &argv);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
    
    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Bind each rank on a node to a GPU (round-robin over the node's devices)
    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);
    MPI_Comm_free(&nodeComm);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices < 1) {
        fprintf(stderr, "No CUDA device available\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    CUDA_CHECK(cudaFree(nullptr));  // establish context outside the timed region

    // Initialize bodies (deterministic, identical on every rank)
    std::vector<Body> bodies(std::max(numBodies, 0));
    randomizeBodies(bodies);

    Simulation sim;
    sim.init(bodies, rank, size);
    CUDA_CHECK(cudaDeviceSynchronize());
    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        sim.step();
    }
    CUDA_CHECK(cudaStreamSynchronize(sim.stream));
    MPI_Barrier(MPI_COMM_WORLD);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    if (rank == 0) printf("Simulation time: %ld ms\n", duration.count());

    if (printResults || validate) sim.gather(bodies);
    
    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData(numBodies * 6);
#pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            bodyData[6 * i + 0] = bodies[i].pos.x;
            bodyData[6 * i + 1] = bodies[i].pos.y;
            bodyData[6 * i + 2] = bodies[i].pos.z;
            bodyData[6 * i + 3] = bodies[i].vel.x;
            bodyData[6 * i + 4] = bodies[i].vel.y;
            bodyData[6 * i + 5] = bodies[i].vel.z;
        }
        print_results(bodyData, "Bodies");
    }
    
    int exitCode = 0;
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        if (rank == 0) printf("Validating simulation results...\n");

        int valid = rank == 0 ? (validateSimulation(bodies) ? 1 : 0) : 0;
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
        if (valid) {
            // Report final energy for reference
            double finalEnergy = sim.totalEnergy();
            if (rank == 0) {
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            }
        } else {
            if (rank == 0) printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    sim.destroy();
    MPI_Finalize();
    return exitCode;
}
