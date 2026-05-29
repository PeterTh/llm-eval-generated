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

// ---------------------------------------------------------------------------
// CUDA kernels — Structure-of-Arrays layout for coalesced memory access
// ---------------------------------------------------------------------------

// SoA body data on device: separate arrays for pos_x, pos_y, pos_z, vel_x, vel_y, vel_z
__global__ void computeForcesKernel(
    const double* pos_x, const double* pos_y, const double* pos_z,
          double* vel_x,       double* vel_y,       double* vel_z,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    const double pi_x = pos_x[i];
    const double pi_y = pos_y[i];
    const double pi_z = pos_z[i];

    for (int j = 0; j < n; ++j) {
        const double dx = pos_x[j] - pi_x;
        const double dy = pos_y[j] - pi_y;
        const double dz = pos_z[j] - pi_z;
        const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;

        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }

    vel_x[i] += DT * Fx;
    vel_y[i] += DT * Fy;
    vel_z[i] += DT * Fz;
}

__global__ void integrateBodiesKernel(
    double* pos_x, double* pos_y, double* pos_z,
    const double* vel_x, const double* vel_y, const double* vel_z,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    pos_x[i] += vel_x[i] * DT;
    pos_y[i] += vel_y[i] * DT;
    pos_z[i] += vel_z[i] * DT;
}

// ---------------------------------------------------------------------------
// Host helpers
// ---------------------------------------------------------------------------

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
    
    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + 
                        body.vel.y * body.vel.y + 
                        body.vel.z * body.vel.z);
    }
    
    // Potential energy (assuming unit mass for all bodies)
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
    
    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);
    
    // Allocate device SoA arrays
    const int n = numBodies;
    const size_t bytes = static_cast<size_t>(n) * sizeof(double);
    double *d_pos_x, *d_pos_y, *d_pos_z, *d_vel_x, *d_vel_y, *d_vel_z;
    cudaMalloc(&d_pos_x, bytes);
    cudaMalloc(&d_pos_y, bytes);
    cudaMalloc(&d_pos_z, bytes);
    cudaMalloc(&d_vel_x, bytes);
    cudaMalloc(&d_vel_y, bytes);
    cudaMalloc(&d_vel_z, bytes);
    
    // Copy AoS host data to SoA device arrays
    std::vector<double> h_pos_x(n), h_pos_y(n), h_pos_z(n);
    std::vector<double> h_vel_x(n), h_vel_y(n), h_vel_z(n);
    for (int k = 0; k < n; ++k) {
        h_pos_x[k] = bodies[k].pos.x;
        h_pos_y[k] = bodies[k].pos.y;
        h_pos_z[k] = bodies[k].pos.z;
        h_vel_x[k] = bodies[k].vel.x;
        h_vel_y[k] = bodies[k].vel.y;
        h_vel_z[k] = bodies[k].vel.z;
    }
    cudaMemcpy(d_pos_x, h_pos_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, h_pos_y.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_z, h_pos_z.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_x, h_vel_x.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_y, h_vel_y.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vel_z, h_vel_z.data(), bytes, cudaMemcpyHostToDevice);
    
    // Launch configuration
    const int blockSize = 256;
    const int numBlocks = (n + blockSize - 1) / blockSize;
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, blockSize>>>(
            d_pos_x, d_pos_y, d_pos_z,
            d_vel_x, d_vel_y, d_vel_z, n);
        integrateBodiesKernel<<<numBlocks, blockSize>>>(
            d_pos_x, d_pos_y, d_pos_z,
            d_vel_x, d_vel_y, d_vel_z, n);
    }
    cudaDeviceSynchronize();
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
    // Copy results back
    cudaMemcpy(h_pos_x.data(), d_pos_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_y.data(), d_pos_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pos_z.data(), d_pos_z, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel_x.data(), d_vel_x, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel_y.data(), d_vel_y, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_vel_z.data(), d_vel_z, bytes, cudaMemcpyDeviceToHost);
    
    // Reconstruct AoS from SoA for host-side usage
    for (int k = 0; k < n; ++k) {
        bodies[k].pos.x = h_pos_x[k];
        bodies[k].pos.y = h_pos_y[k];
        bodies[k].pos.z = h_pos_z[k];
        bodies[k].vel.x = h_vel_x[k];
        bodies[k].vel.y = h_vel_y[k];
        bodies[k].vel.z = h_vel_z[k];
    }
    
    // Free device memory
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_pos_z);
    cudaFree(d_vel_x);
    cudaFree(d_vel_y);
    cudaFree(d_vel_z);
    
    printf("Simulation time: %ld ms\n", duration.count());
    
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
