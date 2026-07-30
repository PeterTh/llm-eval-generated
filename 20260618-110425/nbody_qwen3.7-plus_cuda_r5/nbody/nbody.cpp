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

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define BLOCK_SIZE 256

__global__ void computeForcesKernel(
    const double* __restrict__ posX, const double* __restrict__ posY, const double* __restrict__ posZ,
    double* __restrict__ velX, double* __restrict__ velY, double* __restrict__ velZ,
    const int n)
{
    __shared__ double sposX[BLOCK_SIZE];
    __shared__ double sposY[BLOCK_SIZE];
    __shared__ double sposZ[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double myPosX = 0.0, myPosY = 0.0, myPosZ = 0.0;

    if (i < n) {
        myPosX = posX[i];
        myPosY = posY[i];
        myPosZ = posZ[i];
    }

    const int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int j = tile * BLOCK_SIZE + threadIdx.x;

        sposX[threadIdx.x] = (j < n) ? posX[j] : 0.0;
        sposY[threadIdx.x] = (j < n) ? posY[j] : 0.0;
        sposZ[threadIdx.x] = (j < n) ? posZ[j] : 0.0;

        __syncthreads();

        if (i < n) {
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                if (tile * BLOCK_SIZE + k >= n) break;

                double dx = sposX[k] - myPosX;
                double dy = sposY[k] - myPosY;
                double dz = sposZ[k] - myPosZ;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;

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

__global__ void integrateBodiesKernel(
    double* __restrict__ posX, double* __restrict__ posY, double* __restrict__ posZ,
    const double* __restrict__ velX, const double* __restrict__ velY, const double* __restrict__ velZ,
    const int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        posX[i] += velX[i] * DT;
        posY[i] += velY[i] * DT;
        posZ[i] += velZ[i] * DT;
    }
}

struct DeviceBodies {
    double *posX, *posY, *posZ;
    double *velX, *velY, *velZ;
};

void allocDeviceBodies(DeviceBodies& d, size_t n) {
    CUDA_CHECK(cudaMalloc(&d.posX, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.posY, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.posZ, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.velX, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.velY, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d.velZ, n * sizeof(double)));
}

void freeDeviceBodies(DeviceBodies& d) {
    cudaFree(d.posX);
    cudaFree(d.posY);
    cudaFree(d.posZ);
    cudaFree(d.velX);
    cudaFree(d.velY);
    cudaFree(d.velZ);
}

void copyBodiesToDevice(DeviceBodies& d, const std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    std::vector<double> px(n), py(n), pz(n), vx(n), vy(n), vz(n);
    for (size_t i = 0; i < n; ++i) {
        px[i] = bodies[i].pos.x;
        py[i] = bodies[i].pos.y;
        pz[i] = bodies[i].pos.z;
        vx[i] = bodies[i].vel.x;
        vy[i] = bodies[i].vel.y;
        vz[i] = bodies[i].vel.z;
    }
    CUDA_CHECK(cudaMemcpy(d.posX, px.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.posY, py.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.posZ, pz.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.velX, vx.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.velY, vy.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.velZ, vz.data(), n * sizeof(double), cudaMemcpyHostToDevice));
}

void copyBodiesFromDevice(const DeviceBodies& d, std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    std::vector<double> px(n), py(n), pz(n), vx(n), vy(n), vz(n);
    CUDA_CHECK(cudaMemcpy(px.data(), d.posX, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(py.data(), d.posY, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pz.data(), d.posZ, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx.data(), d.velX, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy.data(), d.velY, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vz.data(), d.velZ, n * sizeof(double), cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos.x = px[i];
        bodies[i].pos.y = py[i];
        bodies[i].pos.z = pz[i];
        bodies[i].vel.x = vx[i];
        bodies[i].vel.y = vy[i];
        bodies[i].vel.z = vz[i];
    }
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

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x +
                        body.vel.y * body.vel.y +
                        body.vel.z * body.vel.z);
    }

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

    printf("N-Body Simulation (CUDA)\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    DeviceBodies d;
    allocDeviceBodies(d, numBodies);
    copyBodiesToDevice(d, bodies);

    const int gridSize = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<gridSize, BLOCK_SIZE>>>(
            d.posX, d.posY, d.posZ,
            d.velX, d.velY, d.velZ,
            numBodies);

        integrateBodiesKernel<<<gridSize, BLOCK_SIZE>>>(
            d.posX, d.posY, d.posZ,
            d.velX, d.velY, d.velZ,
            numBodies);
    }

    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    copyBodiesFromDevice(d, bodies);
    freeDeviceBodies(d);

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

    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(bodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }

    return 0;
}
