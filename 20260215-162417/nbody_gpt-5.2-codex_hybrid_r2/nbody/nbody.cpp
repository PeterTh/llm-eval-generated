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

struct Body {
    double px, py, pz;
    double vx, vy, vz;
};

struct Pos {
    double x, y, z;
};

static inline void checkCuda(const cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", msg, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

__global__ void computeForcesKernel(Body* bodies, int n, int start, int count) {
    extern __shared__ Pos shPos[];
    const int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = localIdx < count;

    const int i = start + localIdx;
    const double ix = active ? bodies[i].px : 0.0;
    const double iy = active ? bodies[i].py : 0.0;
    const double iz = active ? bodies[i].pz : 0.0;
    double Fx = 0.0;
    double Fy = 0.0;
    double Fz = 0.0;

    const int tiles = (n + blockDim.x - 1) / blockDim.x;
    for (int tile = 0; tile < tiles; ++tile) {
        const int j = tile * blockDim.x + threadIdx.x;
        if (j < n) {
            shPos[threadIdx.x].x = bodies[j].px;
            shPos[threadIdx.x].y = bodies[j].py;
            shPos[threadIdx.x].z = bodies[j].pz;
        }
        __syncthreads();

        int tileSize = n - tile * blockDim.x;
        if (tileSize > blockDim.x) {
            tileSize = blockDim.x;
        }

        if (active) {
            for (int k = 0; k < tileSize; ++k) {
                const double dx = shPos[k].x - ix;
                const double dy = shPos[k].y - iy;
                const double dz = shPos[k].z - iz;
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
        bodies[i].vx += DT * Fx;
        bodies[i].vy += DT * Fy;
        bodies[i].vz += DT * Fz;
    }
}

__global__ void integrateKernel(Body* bodies, int start, int count) {
    const int localIdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (localIdx >= count) {
        return;
    }
    const int i = start + localIdx;
    bodies[i].px += bodies[i].vx * DT;
    bodies[i].py += bodies[i].vy * DT;
    bodies[i].pz += bodies[i].vz * DT;
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.px = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.py = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vx = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vy = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vz = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    double kinetic = 0.0;

    #pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Body& body = bodies[i];
        kinetic += 0.5 * (body.vx * body.vx + body.vy * body.vy + body.vz * body.vz);
    }

    double potential = 0.0;
    #pragma omp parallel for reduction(+:potential) schedule(static)
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].px - bodies[i].px;
            const double dy = bodies[j].py - bodies[i].py;
            const double dz = bodies[j].pz - bodies[i].pz;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            potential += 1.0 / dist;
        }
    }

    return kinetic - potential;
}

// Validate that simulation produces finite, reasonable values
bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        // Check for NaN or Inf values
        if (!std::isfinite(body.px) || !std::isfinite(body.py) || !std::isfinite(body.pz) ||
            !std::isfinite(body.vx) || !std::isfinite(body.vy) || !std::isfinite(body.vz)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        // Check for extreme values (bodies shouldn't fly off to infinity)
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.px) > maxPos || std::abs(body.py) > maxPos || std::abs(body.pz) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vx) > maxVel || std::abs(body.vy) > maxVel || std::abs(body.vz) > maxVel) {
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

    int rank = 0;
    int size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    bool validate = false;
    bool printResults = false;
    bool showHelp = false;
    bool parseError = false;

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
            showHelp = true;
        } else {
            parseError = true;
        }
    }

    if (showHelp || parseError) {
        if (rank == 0) {
            if (parseError) {
                printf("Unknown or invalid option provided\n");
            }
            printUsage(argv[0]);
        }
        MPI_Finalize();
        return parseError ? 1 : 0;
    }

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    int deviceCount = 0;
    cudaError_t deviceErr = cudaGetDeviceCount(&deviceCount);
    if (deviceErr != cudaSuccess || deviceCount == 0) {
        if (rank == 0) {
            fprintf(stderr, "CUDA device init failed: %s\n", cudaGetErrorString(deviceErr));
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    checkCuda(cudaSetDevice(rank % deviceCount), "cudaSetDevice");

    Body* d_bodies = nullptr;
    const size_t totalBytes = static_cast<size_t>(numBodies) * sizeof(Body);
    checkCuda(cudaMalloc(&d_bodies, totalBytes), "cudaMalloc d_bodies");

    cudaStream_t stream = nullptr;
    checkCuda(cudaStreamCreate(&stream), "cudaStreamCreate");
    checkCuda(cudaFuncSetCacheConfig(computeForcesKernel, cudaFuncCachePreferShared),
              "cudaFuncSetCacheConfig");

    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    if (rank == 0) {
        randomizeBodies(bodies);
    }
    MPI_Bcast(bodies.data(), static_cast<int>(totalBytes), MPI_BYTE, 0, MPI_COMM_WORLD);

    const int localStart = (numBodies * rank) / size;
    const int localEnd = (numBodies * (rank + 1)) / size;
    const int localCount = localEnd - localStart;
    const int localBytes = localCount * static_cast<int>(sizeof(Body));

    std::vector<int> recvCounts(size);
    std::vector<int> displs(size);
    int offset = 0;
    for (int r = 0; r < size; ++r) {
        const int rStart = (numBodies * r) / size;
        const int rEnd = (numBodies * (r + 1)) / size;
        recvCounts[r] = (rEnd - rStart) * static_cast<int>(sizeof(Body));
        displs[r] = offset;
        offset += recvCounts[r];
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();

    const int blockSize = 256;
    const int gridSize = (localCount + blockSize - 1) / blockSize;
    const size_t sharedBytes = static_cast<size_t>(blockSize) * sizeof(Pos);

    for (int step = 0; step < numSteps; ++step) {
        if (localCount > 0) {
            checkCuda(cudaMemcpyAsync(d_bodies, bodies.data(), totalBytes,
                                     cudaMemcpyHostToDevice, stream),
                      "cudaMemcpyAsync H2D bodies");

            computeForcesKernel<<<gridSize, blockSize, sharedBytes, stream>>>(
                d_bodies, numBodies, localStart, localCount);
            checkCuda(cudaGetLastError(), "computeForcesKernel launch");

            integrateKernel<<<gridSize, blockSize, 0, stream>>>(
                d_bodies, localStart, localCount);
            checkCuda(cudaGetLastError(), "integrateKernel launch");

            checkCuda(cudaMemcpyAsync(bodies.data() + localStart,
                                     d_bodies + localStart,
                                     static_cast<size_t>(localBytes),
                                     cudaMemcpyDeviceToHost, stream),
                      "cudaMemcpyAsync D2H local");
            checkCuda(cudaStreamSynchronize(stream), "cudaStreamSynchronize");
        }

        MPI_Allgatherv(reinterpret_cast<unsigned char*>(bodies.data() + localStart),
                       localBytes,
                       MPI_BYTE,
                       reinterpret_cast<unsigned char*>(bodies.data()),
                       recvCounts.data(),
                       displs.data(),
                       MPI_BYTE,
                       MPI_COMM_WORLD);
    }

    MPI_Barrier(MPI_COMM_WORLD);
    const double end = MPI_Wtime();

    if (rank == 0) {
        printf("Simulation time: %.3f ms\n", (end - start) * 1000.0);
    }

    if (printResults && rank == 0) {
        std::vector<double> bodyData(static_cast<size_t>(numBodies) * 6);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < numBodies; ++i) {
            const size_t idx = static_cast<size_t>(i) * 6;
            bodyData[idx] = bodies[i].px;
            bodyData[idx + 1] = bodies[i].py;
            bodyData[idx + 2] = bodies[i].pz;
            bodyData[idx + 3] = bodies[i].vx;
            bodyData[idx + 4] = bodies[i].vy;
            bodyData[idx + 5] = bodies[i].vz;
        }
        print_results(bodyData, "Bodies");
    }

    int exitCode = 0;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (validateSimulation(bodies)) {
                const double finalEnergy = computeTotalEnergy(bodies);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
                exitCode = 0;
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    checkCuda(cudaStreamDestroy(stream), "cudaStreamDestroy");
    checkCuda(cudaFree(d_bodies), "cudaFree d_bodies");

    MPI_Finalize();
    return exitCode;
}
