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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static inline void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s\n", msg, cudaGetErrorString(err));
        std::exit(1);
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

// CUDA kernels and wrappers - unconditionally use CUDA for parallelism

__global__ void computeForcesKernel(const double* __restrict__ px, const double* __restrict__ py, const double* __restrict__ pz,
                                    double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
                                    int n) {
    extern __shared__ double sdata[]; // sx[blockDim.x], sy[blockDim.x], sz[blockDim.x]
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    if (i >= n) return;

    const double xi = px[i];
    const double yi = py[i];
    const double zi = pz[i];

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    const int blockSize = blockDim.x;
    double* sx = sdata;
    double* sy = sdata + blockSize;
    double* sz = sdata + 2 * blockSize;

    const int tiles = (n + blockSize - 1) / blockSize;
    for (int t = 0; t < tiles; ++t) {
        int idx = t * blockSize + tid;
        if (idx < n) {
            sx[tid] = px[idx];
            sy[tid] = py[idx];
            sz[tid] = pz[idx];
        } else {
            sx[tid] = 0.0;
            sy[tid] = 0.0;
            sz[tid] = 0.0;
        }
        __syncthreads();

        int limit = n - t * blockSize;
        if (limit > blockSize) limit = blockSize;
        for (int j = 0; j < limit; ++j) {
            const double dx = sx[j] - xi;
            const double dy = sy[j] - yi;
            const double dz = sz[j] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = rsqrt(distSqr); // rsqrt for double isn't intrinsic; use 1/sqrt
            const double inv = 1.0 / sqrt(distSqr);
            const double invDist3 = inv * inv * inv;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }

    vx[i] += DT * Fx;
    vy[i] += DT * Fy;
    vz[i] += DT * Fz;
}

__global__ void integrateKernel(double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
                                const double* __restrict__ vx, const double* __restrict__ vy, const double* __restrict__ vz,
                                int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    px[i] += vx[i] * DT;
    py[i] += vy[i] * DT;
    pz[i] += vz[i] * DT;
}

void computeForcesCUDA(std::vector<Body>& bodies) {
    const int n = (int)bodies.size();
    size_t bytes = n * sizeof(double);

    double* h_px = (double*)malloc(bytes);
    double* h_py = (double*)malloc(bytes);
    double* h_pz = (double*)malloc(bytes);
    double* h_vx = (double*)malloc(bytes);
    double* h_vy = (double*)malloc(bytes);
    double* h_vz = (double*)malloc(bytes);

    for (int i = 0; i < n; ++i) {
        h_px[i] = bodies[i].pos.x;
        h_py[i] = bodies[i].pos.y;
        h_pz[i] = bodies[i].pos.z;
        h_vx[i] = bodies[i].vel.x;
        h_vy[i] = bodies[i].vel.y;
        h_vz[i] = bodies[i].vel.z;
    }

    double *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
    checkCuda(cudaMalloc((void**)&d_px, bytes), "cudaMalloc d_px");
    checkCuda(cudaMalloc((void**)&d_py, bytes), "cudaMalloc d_py");
    checkCuda(cudaMalloc((void**)&d_pz, bytes), "cudaMalloc d_pz");
    checkCuda(cudaMalloc((void**)&d_vx, bytes), "cudaMalloc d_vx");
    checkCuda(cudaMalloc((void**)&d_vy, bytes), "cudaMalloc d_vy");
    checkCuda(cudaMalloc((void**)&d_vz, bytes), "cudaMalloc d_vz");

    checkCuda(cudaMemcpy(d_px, h_px, bytes, cudaMemcpyHostToDevice), "memcpy px to device");
    checkCuda(cudaMemcpy(d_py, h_py, bytes, cudaMemcpyHostToDevice), "memcpy py to device");
    checkCuda(cudaMemcpy(d_pz, h_pz, bytes, cudaMemcpyHostToDevice), "memcpy pz to device");
    checkCuda(cudaMemcpy(d_vx, h_vx, bytes, cudaMemcpyHostToDevice), "memcpy vx to device");
    checkCuda(cudaMemcpy(d_vy, h_vy, bytes, cudaMemcpyHostToDevice), "memcpy vy to device");
    checkCuda(cudaMemcpy(d_vz, h_vz, bytes, cudaMemcpyHostToDevice), "memcpy vz to device");

    const int blockSize = 256;
    const int gridSize = (n + blockSize - 1) / blockSize;
    const size_t sharedBytes = 3 * blockSize * sizeof(double);

    // compute forces and integrate on device
    computeForcesKernel<<<gridSize, blockSize, sharedBytes>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, n);
    checkCuda(cudaGetLastError(), "computeForcesKernel launch");
    integrateKernel<<<gridSize, blockSize>>>(d_px, d_py, d_pz, d_vx, d_vy, d_vz, n);
    checkCuda(cudaGetLastError(), "integrateKernel launch");

    checkCuda(cudaMemcpy(h_px, d_px, bytes, cudaMemcpyDeviceToHost), "memcpy px to host");
    checkCuda(cudaMemcpy(h_py, d_py, bytes, cudaMemcpyDeviceToHost), "memcpy py to host");
    checkCuda(cudaMemcpy(h_pz, d_pz, bytes, cudaMemcpyDeviceToHost), "memcpy pz to host");
    checkCuda(cudaMemcpy(h_vx, d_vx, bytes, cudaMemcpyDeviceToHost), "memcpy vx to host");
    checkCuda(cudaMemcpy(h_vy, d_vy, bytes, cudaMemcpyDeviceToHost), "memcpy vy to host");
    checkCuda(cudaMemcpy(h_vz, d_vz, bytes, cudaMemcpyDeviceToHost), "memcpy vz to host");

    // copy back
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = h_px[i];
        bodies[i].pos.y = h_py[i];
        bodies[i].pos.z = h_pz[i];
        bodies[i].vel.x = h_vx[i];
        bodies[i].vel.y = h_vy[i];
        bodies[i].vel.z = h_vz[i];
    }

    // free
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz); cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    free(h_px); free(h_py); free(h_pz); free(h_vx); free(h_vy); free(h_vz);
}

// Host wrappers now call CUDA implementation
void computeForces(std::vector<Body>& bodies) {
    computeForcesCUDA(bodies);
}

void integrateBodies(std::vector<Body>& bodies) {
    // integration is performed on device immediately after forces in computeForcesCUDA,
    // so this is a no-op to preserve original semantics
}

// The remaining functions (computeTotalEnergy, validateSimulation, printUsage, main) remain logically the same

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

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
        integrateBodies(bodies);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

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
