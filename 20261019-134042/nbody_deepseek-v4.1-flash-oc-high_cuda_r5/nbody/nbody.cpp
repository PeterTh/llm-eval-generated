#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// One warp per block maximizes the number of blocks (and thus SM coverage) for
// small body counts while remaining compute-bound for large ones.
constexpr int BLOCK_SIZE = 32;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                        \
    do {                                                                        \
        cudaError_t err__ = (call);                                             \
        if (err__ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error %s at %s:%d\n",                         \
                    cudaGetErrorString(err__), __FILE__, __LINE__);             \
            std::exit(1);                                                       \
        }                                                                       \
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

// Device buffers, structure-of-arrays layout for coalesced access.
struct DeviceBodies {
    double* posX[2] = {nullptr, nullptr};
    double* posY[2] = {nullptr, nullptr};
    double* posZ[2] = {nullptr, nullptr};
    double* velX = nullptr;
    double* velY = nullptr;
    double* velZ = nullptr;
    int n = 0;
};

static void allocDeviceBodies(DeviceBodies& d, int n) {
    d.n = n;
    if (n <= 0) {
        return;
    }
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d.posX[b], bytes));
        CUDA_CHECK(cudaMalloc(&d.posY[b], bytes));
        CUDA_CHECK(cudaMalloc(&d.posZ[b], bytes));
    }
    CUDA_CHECK(cudaMalloc(&d.velX, bytes));
    CUDA_CHECK(cudaMalloc(&d.velY, bytes));
    CUDA_CHECK(cudaMalloc(&d.velZ, bytes));
}

static void freeDeviceBodies(DeviceBodies& d) {
    for (int b = 0; b < 2; ++b) {
        if (d.posX[b]) cudaFree(d.posX[b]);
        if (d.posY[b]) cudaFree(d.posY[b]);
        if (d.posZ[b]) cudaFree(d.posZ[b]);
        d.posX[b] = d.posY[b] = d.posZ[b] = nullptr;
    }
    if (d.velX) cudaFree(d.velX);
    if (d.velY) cudaFree(d.velY);
    if (d.velZ) cudaFree(d.velZ);
    d.velX = d.velY = d.velZ = nullptr;
}

static void uploadBodies(DeviceBodies& d, const std::vector<Body>& bodies) {
    if (d.n <= 0) {
        return;
    }
    std::vector<double> px(d.n), py(d.n), pz(d.n), vx(d.n), vy(d.n), vz(d.n);
    for (int i = 0; i < d.n; ++i) {
        px[i] = bodies[i].pos.x;
        py[i] = bodies[i].pos.y;
        pz[i] = bodies[i].pos.z;
        vx[i] = bodies[i].vel.x;
        vy[i] = bodies[i].vel.y;
        vz[i] = bodies[i].vel.z;
    }
    const size_t bytes = static_cast<size_t>(d.n) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d.posX[0], px.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.posY[0], py.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.posZ[0], pz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.velX, vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.velY, vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.velZ, vz.data(), bytes, cudaMemcpyHostToDevice));
}

static void downloadBodies(const DeviceBodies& d, const double* posX, const double* posY,
                           const double* posZ, std::vector<Body>& bodies) {
    if (d.n <= 0) {
        return;
    }
    std::vector<double> px(d.n), py(d.n), pz(d.n), vx(d.n), vy(d.n), vz(d.n);
    const size_t bytes = static_cast<size_t>(d.n) * sizeof(double);
    CUDA_CHECK(cudaMemcpy(px.data(), posX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(py.data(), posY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pz.data(), posZ, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx.data(), d.velX, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy.data(), d.velY, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vz.data(), d.velZ, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < d.n; ++i) {
        bodies[i].pos = Vec3(px[i], py[i], pz[i]);
        bodies[i].vel = Vec3(vx[i], vy[i], vz[i]);
    }
}

// One simulation step: forces from the source positions, velocity update and
// position integration. New positions go to the destination buffer so the
// source can be read by every block without a race, fusing integrate into the
// same pass and halving kernel launches.
__global__ __launch_bounds__(BLOCK_SIZE)
void nbodyStepKernel(const double* __restrict__ srcX,
                     const double* __restrict__ srcY,
                     const double* __restrict__ srcZ,
                     double* __restrict__ dstX,
                     double* __restrict__ dstY,
                     double* __restrict__ dstZ,
                     double* __restrict__ velX,
                     double* __restrict__ velY,
                     double* __restrict__ velZ,
                     int n) {
    __shared__ double sX[BLOCK_SIZE];
    __shared__ double sY[BLOCK_SIZE];
    __shared__ double sZ[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * BLOCK_SIZE + tid;
    const bool active = i < n;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (active) {
        xi = srcX[i];
        yi = srcY[i];
        zi = srcZ[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    const int fullTiles = (n / BLOCK_SIZE) * BLOCK_SIZE;
    for (int tile = 0; tile < fullTiles; tile += BLOCK_SIZE) {
        const int j = tile + tid;
        sX[tid] = srcX[j];
        sY[tid] = srcY[j];
        sZ[tid] = srcZ[j];
        __syncthreads();

        #pragma unroll 16
        for (int k = 0; k < BLOCK_SIZE; ++k) {
            const double dx = sX[k] - xi;
            const double dy = sY[k] - yi;
            const double dz = sZ[k] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    const int remainder = n - fullTiles;
    if (remainder > 0) {
        const int j = fullTiles + tid;
        if (j < n) {
            sX[tid] = srcX[j];
            sY[tid] = srcY[j];
            sZ[tid] = srcZ[j];
        } else {
            // Self position yields zero displacement and therefore zero force.
            sX[tid] = xi;
            sY[tid] = yi;
            sZ[tid] = zi;
        }
        __syncthreads();

        for (int k = 0; k < remainder; ++k) {
            const double dx = sX[k] - xi;
            const double dy = sY[k] - yi;
            const double dz = sZ[k] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
    }

    if (active) {
        const double vx = velX[i] + DT * Fx;
        const double vy = velY[i] + DT * Fy;
        const double vz = velZ[i] + DT * Fz;
        velX[i] = vx;
        velY[i] = vy;
        velZ[i] = vz;
        dstX[i] = xi + vx * DT;
        dstY[i] = yi + vy * DT;
        dstZ[i] = zi + vz * DT;
    }
}

__global__ __launch_bounds__(BLOCK_SIZE)
void energyKernel(const double* __restrict__ posX,
                  const double* __restrict__ posY,
                  const double* __restrict__ posZ,
                  const double* __restrict__ velX,
                  const double* __restrict__ velY,
                  const double* __restrict__ velZ,
                  double* __restrict__ result,
                  int n) {
    __shared__ double partial[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * BLOCK_SIZE + tid;
    double energy = 0.0;

    if (i < n) {
        const double vx = velX[i];
        const double vy = velY[i];
        const double vz = velZ[i];
        energy += 0.5 * (vx * vx + vy * vy + vz * vz);

        const double xi = posX[i];
        const double yi = posY[i];
        const double zi = posZ[i];
        for (int j = i + 1; j < n; ++j) {
            const double dx = posX[j] - xi;
            const double dy = posY[j] - yi;
            const double dz = posZ[j] - zi;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    partial[tid] = energy;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            partial[tid] += partial[tid + stride];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(result, partial[0]);
    }
}

static double computeTotalEnergy(const DeviceBodies& d, const double* posX,
                                 const double* posY, const double* posZ) {
    double* devResult = nullptr;
    CUDA_CHECK(cudaMalloc(&devResult, sizeof(double)));
    CUDA_CHECK(cudaMemset(devResult, 0, sizeof(double)));
    const int blocks = (d.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    energyKernel<<<blocks, BLOCK_SIZE>>>(posX, posY, posZ, d.velX, d.velY, d.velZ,
                                         devResult, d.n);
    CUDA_CHECK(cudaGetLastError());
    double energy = 0.0;
    CUDA_CHECK(cudaMemcpy(&energy, devResult, sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(devResult));
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
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }
    
    if (numBodies < 0) {
        numBodies = 0;
    }
    
    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies on the host (identical RNG sequence to the reference)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    DeviceBodies device;
    allocDeviceBodies(device, numBodies);
    uploadBodies(device, bodies);

    const int blocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int cur = 0;
    int next = 1;

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps && numBodies > 0; ++step) {
        nbodyStepKernel<<<blocks, BLOCK_SIZE>>>(
            device.posX[cur], device.posY[cur], device.posZ[cur],
            device.posX[next], device.posY[next], device.posZ[next],
            device.velX, device.velY, device.velZ, numBodies);
        const int swap = cur;
        cur = next;
        next = swap;
    }
    CUDA_CHECK(cudaGetLastError());
    if (numBodies > 0) {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    printf("Simulation time: %ld ms\n", duration.count());

    double finalEnergy = 0.0;
    if (validate && numBodies > 0) {
        finalEnergy = computeTotalEnergy(device, device.posX[cur], device.posY[cur], device.posZ[cur]);
    }

    downloadBodies(device, device.posX[cur], device.posY[cur], device.posZ[cur], bodies);
    
    // Print results for external validation
    if (printResults) {
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
    
    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");
        
        if (validateSimulation(bodies)) {
            // Report final energy for reference
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            freeDeviceBodies(device);
            return 0;
        } else {
            printf("Validation: FAILED\n");
            freeDeviceBodies(device);
            return 1;
        }
    }
    
    freeDeviceBodies(device);
    return 0;
}
