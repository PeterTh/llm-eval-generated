#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

#define BLOCK_SIZE 256
#define TILE_DIM 256

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept
        : x(x), y(y), z(z) {}
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
// CUDA Kernels
// ---------------------------------------------------------------------------

// Force computation with shared-memory tiling.
// Each thread computes the total force on one body and updates its velocity.
__global__ __launch_bounds__(BLOCK_SIZE)
void computeForcesKernel(
    const double* __restrict__ posX,
    const double* __restrict__ posY,
    const double* __restrict__ posZ,
    double* __restrict__ velX,
    double* __restrict__ velY,
    double* __restrict__ velZ,
    const int n)
{
    __shared__ double sPosx[TILE_DIM];
    __shared__ double sPosy[TILE_DIM];
    __shared__ double sPosz[TILE_DIM];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double myPosX = 0.0, myPosY = 0.0, myPosZ = 0.0;

    if (i < n) {
        myPosX = posX[i];
        myPosY = posY[i];
        myPosZ = posZ[i];
    }

    const int numTiles = (n + TILE_DIM - 1) / TILE_DIM;

    for (int tile = 0; tile < numTiles; ++tile) {
        const int j = tile * TILE_DIM + threadIdx.x;
        if (j < n) {
            sPosx[threadIdx.x] = posX[j];
            sPosy[threadIdx.x] = posY[j];
            sPosz[threadIdx.x] = posZ[j];
        }
        __syncthreads();

        if (i < n) {
            const int tileEnd = min(TILE_DIM, n - tile * TILE_DIM);
            #pragma unroll 8
            for (int jj = 0; jj < tileEnd; ++jj) {
                const double dx = sPosx[jj] - myPosX;
                const double dy = sPosy[jj] - myPosY;
                const double dz = sPosz[jj] - myPosZ;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = rsqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        velX[i] += DT * Fx;
        velY[i] += DT * Fy;
        velZ[i] += DT * Fz;
    }
}

// Position integration: one thread per body.
__global__ __launch_bounds__(BLOCK_SIZE)
void integrateBodiesKernel(
    double* __restrict__ posX,
    double* __restrict__ posY,
    double* __restrict__ posZ,
    const double* __restrict__ velX,
    const double* __restrict__ velY,
    const double* __restrict__ velZ,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        posX[i] += velX[i] * DT;
        posY[i] += velY[i] * DT;
        posZ[i] += velZ[i] * DT;
    }
}

// Total energy (kinetic + potential) with block-level parallel reduction.
// Each thread i handles body i: kinetic for body i + potential for all pairs (i, j>i).
__global__ __launch_bounds__(BLOCK_SIZE)
void computeEnergyKernel(
    const double* __restrict__ posX,
    const double* __restrict__ posY,
    const double* __restrict__ posZ,
    const double* __restrict__ velX,
    const double* __restrict__ velY,
    const double* __restrict__ velZ,
    double* __restrict__ partialSums,
    const int n)
{
    __shared__ double sdata[BLOCK_SIZE];
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;

    double e = 0.0;

    if (i < n) {
        // Kinetic energy (unit mass)
        e += 0.5 * (velX[i] * velX[i] + velY[i] * velY[i] + velZ[i] * velZ[i]);

        // Potential energy: pairs (i, j) with j > i
        const double px = posX[i], py = posY[i], pz = posZ[i];
        for (int j = i + 1; j < n; ++j) {
            const double dx = posX[j] - px;
            const double dy = posY[j] - py;
            const double dz = posZ[j] - pz;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            e -= 1.0 / dist;
        }
    }

    sdata[tid] = e;
    __syncthreads();

    // Block-level reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) partialSums[blockIdx.x] = sdata[0];
}

// Final reduction across block partial sums.
__global__ void finalReduceKernel(
    const double* __restrict__ partialSums,
    double* __restrict__ result,
    const int numBlocks)
{
    __shared__ double sdata[BLOCK_SIZE];
    const int tid = threadIdx.x;

    double sum = 0.0;
    for (int i = tid; i < numBlocks; i += blockDim.x) {
        sum += partialSums[i];
    }
    sdata[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) result[0] = sdata[0];
}

// ---------------------------------------------------------------------------
// CPU-side helpers (unchanged semantics)
// ---------------------------------------------------------------------------

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
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

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

int main(int argc, char** argv) {
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    const int n = numBodies;

    // Initialize bodies on CPU
    std::vector<Body> bodies(n);
    randomizeBodies(bodies);

    // Deinterleave AoS -> SoA for coalesced GPU access
    std::vector<double> h_posX(n), h_posY(n), h_posZ(n);
    std::vector<double> h_velX(n), h_velY(n), h_velZ(n);
    for (int i = 0; i < n; ++i) {
        h_posX[i] = bodies[i].pos.x;
        h_posY[i] = bodies[i].pos.y;
        h_posZ[i] = bodies[i].pos.z;
        h_velX[i] = bodies[i].vel.x;
        h_velY[i] = bodies[i].vel.y;
        h_velZ[i] = bodies[i].vel.z;
    }

    // Allocate device memory (SoA layout)
    double *d_posX, *d_posY, *d_posZ, *d_velX, *d_velY, *d_velZ;
    CUDA_CHECK(cudaMalloc(&d_posX, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posY, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_posZ, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velX, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velY, (size_t)n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_velZ, (size_t)n * sizeof(double)));

    // Host -> Device
    CUDA_CHECK(cudaMemcpy(d_posX, h_posX.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posY, h_posY.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_posZ, h_posZ.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velX, h_velX.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velY, h_velY.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_velZ, h_velZ.data(), (size_t)n * sizeof(double), cudaMemcpyHostToDevice));

    const int blockSize = BLOCK_SIZE;
    const int gridSize = (n + blockSize - 1) / blockSize;

    // ---- Timed simulation loop (entirely on GPU) ----
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<gridSize, blockSize>>>(
            d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, n);
        integrateBodiesKernel<<<gridSize, blockSize>>>(
            d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, n);
    }

    // If validation requested, compute energy on GPU before copying back
    double finalEnergy = 0.0;
    double* d_partialSums = nullptr;
    double* d_energy = nullptr;

    if (validate && n > 0) {
        CUDA_CHECK(cudaMalloc(&d_partialSums, (size_t)gridSize * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_energy, sizeof(double)));

        computeEnergyKernel<<<gridSize, blockSize>>>(
            d_posX, d_posY, d_posZ, d_velX, d_velY, d_velZ, d_partialSums, n);
        finalReduceKernel<<<1, blockSize>>>(d_partialSums, d_energy, gridSize);

        CUDA_CHECK(cudaMemcpy(&finalEnergy, d_energy, sizeof(double), cudaMemcpyDeviceToHost));
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Device -> Host
    CUDA_CHECK(cudaMemcpy(h_posX.data(), d_posX, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_posY.data(), d_posY, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_posZ.data(), d_posZ, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velX.data(), d_velX, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velY.data(), d_velY, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_velZ.data(), d_velZ, (size_t)n * sizeof(double), cudaMemcpyDeviceToHost));

    // Interleave SoA -> AoS
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = h_posX[i];
        bodies[i].pos.y = h_posY[i];
        bodies[i].pos.z = h_posZ[i];
        bodies[i].vel.x = h_velX[i];
        bodies[i].vel.y = h_velY[i];
        bodies[i].vel.z = h_velZ[i];
    }

    // Print results for external validation
    if (printResults) {
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

    // Validation
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
        }
    }

    // Cleanup
    if (d_partialSums) CUDA_CHECK(cudaFree(d_partialSums));
    if (d_energy) CUDA_CHECK(cudaFree(d_energy));
    CUDA_CHECK(cudaFree(d_posX));
    CUDA_CHECK(cudaFree(d_posY));
    CUDA_CHECK(cudaFree(d_posZ));
    CUDA_CHECK(cudaFree(d_velX));
    CUDA_CHECK(cudaFree(d_velY));
    CUDA_CHECK(cudaFree(d_velZ));

    return 0;
}
