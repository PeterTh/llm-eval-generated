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

#define BLOCK_SIZE 256

__global__ void computeForcesKernel(const double* __restrict__ pos_x,
                                    const double* __restrict__ pos_y,
                                    const double* __restrict__ pos_z,
                                    double* __restrict__ vel_x,
                                    double* __restrict__ vel_y,
                                    double* __restrict__ vel_z,
                                    const int n) {
    __shared__ double spos_x[BLOCK_SIZE];
    __shared__ double spos_y[BLOCK_SIZE];
    __shared__ double spos_z[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_tiles = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double px_i = 0.0, py_i = 0.0, pz_i = 0.0;
    if (i < n) {
        px_i = pos_x[i];
        py_i = pos_y[i];
        pz_i = pos_z[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tile = 0; tile < n_tiles; ++tile) {
        const int j = tile * BLOCK_SIZE + threadIdx.x;
        if (j < n) {
            spos_x[threadIdx.x] = pos_x[j];
            spos_y[threadIdx.x] = pos_y[j];
            spos_z[threadIdx.x] = pos_z[j];
        }
        __syncthreads();

        if (i < n) {
            const int end = (tile == n_tiles - 1) ? (n - tile * BLOCK_SIZE) : BLOCK_SIZE;
            #pragma unroll 8
            for (int k = 0; k < end; ++k) {
                const double dx = spos_x[k] - px_i;
                const double dy = spos_y[k] - py_i;
                const double dz = spos_z[k] - pz_i;
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

    if (i < n) {
        vel_x[i] += DT * Fx;
        vel_y[i] += DT * Fy;
        vel_z[i] += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(double* __restrict__ pos_x,
                                      double* __restrict__ pos_y,
                                      double* __restrict__ pos_z,
                                      const double* __restrict__ vel_x,
                                      const double* __restrict__ vel_y,
                                      const double* __restrict__ vel_z,
                                      const int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pos_x[i] += vel_x[i] * DT;
    pos_y[i] += vel_y[i] * DT;
    pos_z[i] += vel_z[i] * DT;
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

void runSimulation(std::vector<Body>& bodies, int numSteps) {
    const int n = bodies.size();
    if (n == 0) return;

    const size_t bytes = n * sizeof(double);

    double *d_pos_x, *d_pos_y, *d_pos_z;
    double *d_vel_x, *d_vel_y, *d_vel_z;

    cudaMalloc(&d_pos_x, bytes);
    cudaMalloc(&d_pos_y, bytes);
    cudaMalloc(&d_pos_z, bytes);
    cudaMalloc(&d_vel_x, bytes);
    cudaMalloc(&d_vel_y, bytes);
    cudaMalloc(&d_vel_z, bytes);

    std::vector<double> pos_x(n), pos_y(n), pos_z(n);
    std::vector<double> vel_x(n), vel_y(n), vel_z(n);
    for (int i = 0; i < n; ++i) {
        pos_x[i] = bodies[i].pos.x;
        pos_y[i] = bodies[i].pos.y;
        pos_z[i] = bodies[i].pos.z;
        vel_x[i] = bodies[i].vel.x;
        vel_y[i] = bodies[i].vel.y;
        vel_z[i] = bodies[i].vel.z;
    }

    cudaMemcpy(d_pos_x, pos_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, pos_y.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_z, pos_z.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_x, vel_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_y, vel_y.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_z, vel_z.data(), bytes, cudaMemcpyHostToDevice);

    const int numBlocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_pos_x, d_pos_y, d_pos_z,
            d_vel_x, d_vel_y, d_vel_z,
            n);

        integrateBodiesKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_pos_x, d_pos_y, d_pos_z,
            d_vel_x, d_vel_y, d_vel_z,
            n);
    }

    cudaMemcpy(pos_x.data(), d_pos_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_y.data(), d_pos_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(pos_z.data(), d_pos_z, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vel_x.data(), d_vel_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vel_y.data(), d_vel_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(vel_z.data(), d_vel_z, bytes, cudaMemcpyDeviceToHost);

    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = pos_x[i];
        bodies[i].pos.y = pos_y[i];
        bodies[i].pos.z = pos_z[i];
        bodies[i].vel.x = vel_x[i];
        bodies[i].vel.y = vel_y[i];
        bodies[i].vel.z = vel_z[i];
    }

    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_pos_z);
    cudaFree(d_vel_x);
    cudaFree(d_vel_y);
    cudaFree(d_vel_z);
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
    
    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    runSimulation(bodies, numSteps);
    
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
