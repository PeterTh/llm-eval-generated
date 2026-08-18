#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
constexpr int BLOCK_SIZE = 256;

struct Vec3 {
    double x, y, z;
    __host__ __device__ constexpr Vec3(const double x = 0, const double y = 0,
                                       const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

static_assert(sizeof(Body) == 6 * sizeof(double), "Body must remain tightly packed");

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t error = (call);                                                 \
        if (error != cudaSuccess) {                                                       \
            std::fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,       \
                         cudaGetErrorString(error));                                      \
            std::exit(EXIT_FAILURE);                                                      \
        }                                                                                 \
    } while (0)

// Each block evaluates a tile of source positions from shared memory.  Bodies are
// only written by their owning thread, so this has the same force-then-integrate
// ordering as the original implementation without inter-body write races.
__global__ void computeForces(Body* bodies, int n) {
    __shared__ Vec3 tile[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    Vec3 position{};
    double vx = 0.0, vy = 0.0, vz = 0.0;
    if (active) {
        const Body body = bodies[i];
        position = body.pos;
        vx = body.vel.x;
        vy = body.vel.y;
        vz = body.vel.z;
    }

    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int base = 0; base < n; base += BLOCK_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            tile[threadIdx.x] = bodies[j].pos;
        }
        __syncthreads();

        const int count = min(BLOCK_SIZE, n - base);
        if (active) {
            #pragma unroll 8
            for (int k = 0; k < count; ++k) {
                const double dx = tile[k].x - position.x;
                const double dy = tile[k].y - position.y;
                const double dz = tile[k].z - position.z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                fx += dx * invDist3;
                fy += dy * invDist3;
                fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        bodies[i].vel.x = vx + DT * fx;
        bodies[i].vel.y = vy + DT * fy;
        bodies[i].vel.z = vz + DT * fz;
    }
}

__global__ void integrateBodies(Body* bodies, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        Body body = bodies[i];
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
        bodies[i] = body;
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
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            energy -= 1.0 / std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        }
    }
    return energy;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos.x) || !std::isfinite(body.pos.y) || !std::isfinite(body.pos.z) ||
            !std::isfinite(body.vel.x) || !std::isfinite(body.vel.y) || !std::isfinite(body.vel.z)) {
            std::printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        if (std::abs(body.pos.x) > 1e6 || std::abs(body.pos.y) > 1e6 || std::abs(body.pos.z) > 1e6) {
            std::printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel.x) > 1e6 || std::abs(body.vel.y) > 1e6 || std::abs(body.vel.z) > 1e6) {
            std::printf("Validation failed: body velocity exceeds reasonable bounds\n");
            return false;
        }
    }
    return true;
}

void printUsage(const char* progName) {
    std::printf("Usage: %s [options]\n", progName);
    std::printf("Options:\n  -n <num>     Number of bodies (default: 1024)\n"
                "  -s <num>     Number of simulation steps (default: 10)\n"
                "  -v           Enable validation (checks energy conservation)\n"
                "  -r           Print results for external validation\n"
                "  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "-n") == 0 && i + 1 < argc) numBodies = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-s") == 0 && i + 1 < argc) numSteps = std::atoi(argv[++i]);
        else if (std::strcmp(argv[i], "-v") == 0) validate = true;
        else if (std::strcmp(argv[i], "-r") == 0) printResults = true;
        else if (std::strcmp(argv[i], "-h") == 0) { printUsage(argv[0]); return 0; }
        else { std::printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); return 1; }
    }
    if (numBodies < 0 || numSteps < 0) {
        std::fprintf(stderr, "Number of bodies and steps must be non-negative\n");
        return 1;
    }

    std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                numBodies, numSteps, validate ? "enabled" : "disabled");
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    Body* deviceBodies = nullptr;
    const size_t bytes = bodies.size() * sizeof(Body);
    if (bytes != 0) {
        CUDA_CHECK(cudaMalloc(&deviceBodies, bytes));
        CUDA_CHECK(cudaMemcpy(deviceBodies, bodies.data(), bytes, cudaMemcpyHostToDevice));
    }

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));
    CUDA_CHECK(cudaEventRecord(start));
    if (numBodies > 0) {
        const int blocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
        for (int step = 0; step < numSteps; ++step) {
            computeForces<<<blocks, BLOCK_SIZE>>>(deviceBodies, numBodies);
            CUDA_CHECK(cudaGetLastError());
            integrateBodies<<<blocks, BLOCK_SIZE>>>(deviceBodies, numBodies);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));
    float elapsedMs = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&elapsedMs, start, end));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));
    std::printf("Simulation time: %ld ms\n", static_cast<long>(elapsedMs));

    if ((printResults || validate) && bytes != 0)
        CUDA_CHECK(cudaMemcpy(bodies.data(), deviceBodies, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(deviceBodies));

    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(bodies.size() * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos.x); bodyData.push_back(body.pos.y); bodyData.push_back(body.pos.z);
            bodyData.push_back(body.vel.x); bodyData.push_back(body.vel.y); bodyData.push_back(body.vel.z);
        }
        print_results(bodyData, "Bodies");
    }
    if (validate) {
        std::printf("Validating simulation results...\n");
        if (!validateSimulation(bodies)) { std::printf("Validation: FAILED\n"); return 1; }
        std::printf("Final energy: %.6f\nValidation: PASSED\n", computeTotalEnergy(bodies));
    }
    return 0;
}
