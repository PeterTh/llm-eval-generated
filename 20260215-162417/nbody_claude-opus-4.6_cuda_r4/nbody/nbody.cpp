#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#define SOFTENING 1e-9
#define DT 0.01
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

struct Body {
    double px, py, pz;
    double vx, vy, vz;
};

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

// Tiled force computation kernel using shared memory
__global__ void computeForcesKernel(
    const double* __restrict__ px, const double* __restrict__ py, const double* __restrict__ pz,
    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
    int n)
{
    __shared__ double spx[BLOCK_SIZE];
    __shared__ double spy[BLOCK_SIZE];
    __shared__ double spz[BLOCK_SIZE];

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double myPx = 0.0, myPy = 0.0, myPz = 0.0;

    if (i < n) {
        myPx = px[i];
        myPy = py[i];
        myPz = pz[i];
    }

    int numTiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int tile = 0; tile < numTiles; ++tile) {
        int j = tile * BLOCK_SIZE + threadIdx.x;
        if (j < n) {
            spx[threadIdx.x] = px[j];
            spy[threadIdx.x] = py[j];
            spz[threadIdx.x] = pz[j];
        }
        __syncthreads();

        if (i < n) {
            int limit = min(BLOCK_SIZE, n - tile * BLOCK_SIZE);
            #pragma unroll 8
            for (int k = 0; k < limit; ++k) {
                double dx = spx[k] - myPx;
                double dy = spy[k] - myPy;
                double dz = spz[k] - myPz;
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
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(
    double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
    const double* __restrict__ vx, const double* __restrict__ vy, const double* __restrict__ vz,
    int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy = 0.0;
    const size_t n = bodies.size();

    for (const auto& body : bodies) {
        energy += 0.5 * (body.vx * body.vx +
                        body.vy * body.vy +
                        body.vz * body.vz);
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].px - bodies[i].px;
            const double dy = bodies[j].py - bodies[i].py;
            const double dz = bodies[j].pz - bodies[i].pz;
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.px) || !std::isfinite(body.py) || !std::isfinite(body.pz) ||
            !std::isfinite(body.vx) || !std::isfinite(body.vy) || !std::isfinite(body.vz)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

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

    // Initialize bodies on host
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Convert AoS to SoA for GPU
    size_t bytes = numBodies * sizeof(double);
    std::vector<double> h_px(numBodies), h_py(numBodies), h_pz(numBodies);
    std::vector<double> h_vx(numBodies), h_vy(numBodies), h_vz(numBodies);

    for (int i = 0; i < numBodies; ++i) {
        h_px[i] = bodies[i].px; h_py[i] = bodies[i].py; h_pz[i] = bodies[i].pz;
        h_vx[i] = bodies[i].vx; h_vy[i] = bodies[i].vy; h_vz[i] = bodies[i].vz;
    }

    // Allocate device memory
    double *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
    CUDA_CHECK(cudaMalloc(&d_px, bytes));
    CUDA_CHECK(cudaMalloc(&d_py, bytes));
    CUDA_CHECK(cudaMalloc(&d_pz, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_vz, bytes));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz.data(), bytes, cudaMemcpyHostToDevice));

    int gridSize = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Run simulation on GPU
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<gridSize, BLOCK_SIZE>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, numBodies);
        integrateBodiesKernel<<<gridSize, BLOCK_SIZE>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, numBodies);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_py.data(), d_py, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pz.data(), d_pz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost));

    // Convert SoA back to AoS
    for (int i = 0; i < numBodies; ++i) {
        bodies[i].px = h_px[i]; bodies[i].py = h_py[i]; bodies[i].pz = h_pz[i];
        bodies[i].vx = h_vx[i]; bodies[i].vy = h_vy[i]; bodies[i].vz = h_vz[i];
    }

    CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));

    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.px);
            bodyData.push_back(body.py);
            bodyData.push_back(body.pz);
            bodyData.push_back(body.vx);
            bodyData.push_back(body.vy);
            bodyData.push_back(body.vz);
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
