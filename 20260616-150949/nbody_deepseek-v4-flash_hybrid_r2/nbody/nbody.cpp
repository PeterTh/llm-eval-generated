#include <mpi.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(err)); \
        MPI_Abort(MPI_COMM_WORLD, 1); \
    } \
} while(0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

// Device constants
__constant__ double c_softening;
__constant__ double c_dt;

// CUDA kernel: force computation with shared memory tiling
__global__ void computeForcesKernel(
    const double* __restrict__ posX,
    const double* __restrict__ posY,
    const double* __restrict__ posZ,
    double* __restrict__ velX,
    double* __restrict__ velY,
    double* __restrict__ velZ,
    const int n,
    const int localStart,
    const int localEnd)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x + localStart;
    if (i >= localEnd) return;

    const double px = posX[i];
    const double py = posY[i];
    const double pz = posZ[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    const double soft = c_softening;
    const double dt = c_dt;

    // Shared memory for tiled access to body positions
    extern __shared__ double shared[];
    double* sX = &shared[0];
    double* sY = &shared[blockDim.x];
    double* sZ = &shared[2 * blockDim.x];

    const int blockSize = blockDim.x;

    for (int tile = 0; tile < n; tile += blockSize) {
        const int tileEnd = min(tile + blockSize, n);
        const int tileSize = tileEnd - tile;

        // Cooperative load tile into shared memory
        const int tid = threadIdx.x;
        if (tid < tileSize) {
            sX[tid] = posX[tile + tid];
            sY[tid] = posY[tile + tid];
            sZ[tid] = posZ[tile + tid];
        }
        __syncthreads();

        // Compute interactions with bodies in this tile
        #pragma unroll 32
        for (int j = 0; j < tileSize; ++j) {
            const double dx = sX[j] - px;
            const double dy = sY[j] - py;
            const double dz = sZ[j] - pz;
            const double distSqr = dx*dx + dy*dy + dz*dz + soft;
            const double invDist = rsqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    velX[i] += dt * Fx;
    velY[i] += dt * Fy;
    velZ[i] += dt * Fz;
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

    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    int numBodies = 1024;
    int numSteps = 10;
    int validate = 0;
    int printResults = 0;

    // Parse command line arguments on rank 0, broadcast to all ranks
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                MPI_Finalize();
                return 0;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                MPI_Finalize();
                return 1;
            }
        }
    }

    // Broadcast parameters
    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validate, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResults, 1, MPI_INT, 0, MPI_COMM_WORLD);

    // Compute local chunk distribution across MPI ranks
    int base = numBodies / numRanks;
    int rem = numBodies % numRanks;
    int localCount = base + (rank < rem ? 1 : 0);
    int localStart = 0;
    for (int r = 0; r < rank; ++r) {
        localStart += base + (r < rem ? 1 : 0);
    }
    int localEnd = localStart + localCount;

    // Prepare MPI_Allgatherv counts and displacements
    std::vector<int> counts(numRanks), displs(numRanks);
    int disp = 0;
    for (int r = 0; r < numRanks; ++r) {
        int cnt = base + (r < rem ? 1 : 0);
        counts[r] = cnt;
        displs[r] = disp;
        disp += cnt;
    }

    // Set up CUDA device (round-robin assignment across available GPUs)
    int numDevices = 0;
    cudaGetDeviceCount(&numDevices);
    int deviceId = (numDevices > 0) ? (rank % numDevices) : 0;
    CUDA_CHECK(cudaSetDevice(deviceId));

    // Print header (rank 0 only)
    if (rank == 0) {
        printf("N-Body Simulation (Hybrid MPI+OpenMP+CUDA)\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("MPI ranks: %d\n", numRanks);
        printf("CUDA devices: %d\n", numDevices);
        printf("Bodies per rank: %d (avg), %d (this rank)\n", base, localCount);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Allocate host memory in Structure-of-Arrays layout for GPU compatibility
    std::vector<double> h_posX(numBodies), h_posY(numBodies), h_posZ(numBodies);
    std::vector<double> h_velX(numBodies), h_velY(numBodies), h_velZ(numBodies);

    // Initialize bodies on rank 0 (serial, matching original semantics)
    if (rank == 0) {
        unsigned int seed = 42;
        for (int i = 0; i < numBodies; ++i) {
            h_posX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            h_posY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            h_posZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            h_velX[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            h_velY[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
            h_velZ[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        }
    }

    // Broadcast initial data to all MPI ranks
    MPI_Bcast(h_posX.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_posY.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_posZ.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_velX.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_velY.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Bcast(h_velZ.data(), numBodies, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // Allocate device memory (full arrays on each GPU)
    double *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ;
    CUDA_CHECK(cudaMalloc(&d_posX, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posY, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posZ, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velX, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velY, numBodies * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velZ, numBodies * sizeof(double)));

    // Copy initial data host -> device
    CUDA_CHECK(cudaMemcpy(d_posX, h_posX.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, h_posY.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posZ, h_posZ.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, h_velX.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, h_velY.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velZ, h_velZ.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));

    // Set device constants
    double h_soft = SOFTENING;
    double h_dt = DT;
    CUDA_CHECK(cudaMemcpyToSymbol(c_softening, &h_soft, sizeof(double)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_dt, &h_dt, sizeof(double)));

    // CUDA kernel launch configuration
    const int blockSize = 256;
    dim3 blockD(blockSize);
    dim3 gridForce((localCount + blockSize - 1) / blockSize);
    size_t sharedMemSize = 3 * blockSize * sizeof(double);

    MPI_Barrier(MPI_COMM_WORLD);

    // --- Simulation loop ---
    double startTime = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        // Step 1: Compute gravitational forces on GPU (O(n^2) all-pairs)
        //         Each thread handles one body i, computing forces from all j
        //         Updates velocities for this rank's local bodies on device
        computeForcesKernel<<<gridForce, blockD, sharedMemSize>>>(
            d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ,
            numBodies, localStart, localEnd);
        CUDA_CHECK(cudaGetLastError());

        // Step 2: Copy updated velocities (local chunk) from device to host
        CUDA_CHECK(cudaMemcpy(h_velX.data() + localStart, d_velX + localStart,
                               localCount * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_velY.data() + localStart, d_velY + localStart,
                               localCount * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_velZ.data() + localStart, d_velZ + localStart,
                               localCount * sizeof(double), cudaMemcpyDeviceToHost));

        CUDA_CHECK(cudaDeviceSynchronize());

        // Step 3: Integrate positions on CPU with OpenMP (O(n) per rank)
        //         Each body's position update is independent
        #pragma omp parallel for
        for (int i = localStart; i < localEnd; ++i) {
            h_posX[i] += h_velX[i] * h_dt;
            h_posY[i] += h_velY[i] * h_dt;
            h_posZ[i] += h_velZ[i] * h_dt;
        }

        // Step 4: Share updated positions across all MPI ranks
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_posX.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_posY.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                       h_posZ.data(), counts.data(), displs.data(), MPI_DOUBLE,
                       MPI_COMM_WORLD);

        // Step 5: Copy all positions back to GPU for next iteration
        CUDA_CHECK(cudaMemcpy(d_posX, h_posX.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_posY, h_posY.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_posZ, h_posZ.data(), numBodies * sizeof(double), cudaMemcpyHostToDevice));
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    double endTime = MPI_Wtime();

    if (rank == 0) {
        double simTimeMs = (endTime - startTime) * 1000.0;
        printf("Simulation time: %.0f ms\n", simTimeMs);
    }

    // Copy all velocities back to host for validation/results
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   h_velX.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   h_velY.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);
    MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL,
                   h_velZ.data(), counts.data(), displs.data(), MPI_DOUBLE,
                   MPI_COMM_WORLD);

    // Print results for external validation (rank 0 only)
    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(h_posX[i]);
            bodyData.push_back(h_posY[i]);
            bodyData.push_back(h_posZ[i]);
            bodyData.push_back(h_velX[i]);
            bodyData.push_back(h_velY[i]);
            bodyData.push_back(h_velZ[i]);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation on rank 0
    if (validate && rank == 0) {
        printf("Validating simulation results...\n");

        bool valid = true;
        for (int i = 0; i < numBodies; ++i) {
            if (!std::isfinite(h_posX[i]) || !std::isfinite(h_posY[i]) || !std::isfinite(h_posZ[i]) ||
                !std::isfinite(h_velX[i]) || !std::isfinite(h_velY[i]) || !std::isfinite(h_velZ[i])) {
                printf("Validation failed: found NaN or Inf value in body %d\n", i);
                valid = false;
                break;
            }
            const double maxPos = 1e6;
            const double maxVel = 1e6;
            if (std::abs(h_posX[i]) > maxPos || std::abs(h_posY[i]) > maxPos || std::abs(h_posZ[i]) > maxPos) {
                printf("Validation failed: body %d position exceeds reasonable bounds\n", i);
                valid = false;
                break;
            }
            if (std::abs(h_velX[i]) > maxVel || std::abs(h_velY[i]) > maxVel || std::abs(h_velZ[i]) > maxVel) {
                printf("Validation failed: body %d velocity exceeds reasonable bounds\n", i);
                valid = false;
                break;
            }
        }

        if (valid) {
            // Kinetic energy with OpenMP reduction
            double energy = 0.0;
            #pragma omp parallel for reduction(+:energy)
            for (int i = 0; i < numBodies; ++i) {
                energy += 0.5 * (h_velX[i] * h_velX[i] + h_velY[i] * h_velY[i] + h_velZ[i] * h_velZ[i]);
            }
            // Potential energy with OpenMP reduction
            #pragma omp parallel for reduction(+:energy) schedule(guided)
            for (int i = 0; i < numBodies; ++i) {
                for (int j = i + 1; j < numBodies; ++j) {
                    double dx = h_posX[j] - h_posX[i];
                    double dy = h_posY[j] - h_posY[i];
                    double dz = h_posZ[j] - h_posZ[i];
                    double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                    energy -= 1.0 / dist;
                }
            }
            printf("Final energy: %.6f\n", energy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            MPI_Finalize();
            return 1;
        }
    }

    // Cleanup device memory
    CUDA_CHECK(cudaFree(d_posX));
    CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_posZ));
    CUDA_CHECK(cudaFree(d_velX));
    CUDA_CHECK(cudaFree(d_velY));
    CUDA_CHECK(cudaFree(d_velZ));

    MPI_Finalize();
    return 0;
}
