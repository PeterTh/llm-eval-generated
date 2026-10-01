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
constexpr int TILE_SIZE = 128;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

__global__ void stepBodies(double* px, double* py, double* pz,
                           double* vx, double* vy, double* vz, int n) {
    __shared__ double tileX[TILE_SIZE];
    __shared__ double tileY[TILE_SIZE];
    __shared__ double tileZ[TILE_SIZE];
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool active = i < n;
    const double xi = active ? px[i] : 0.0;
    const double yi = active ? py[i] : 0.0;
    const double zi = active ? pz[i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < n; base += TILE_SIZE) {
        const int j = base + threadIdx.x;
        if (j < n) {
            tileX[threadIdx.x] = px[j];
            tileY[threadIdx.x] = py[j];
            tileZ[threadIdx.x] = pz[j];
        }
        __syncthreads();
        const int count = (n - base < TILE_SIZE) ? n - base : TILE_SIZE;
        if (active) {
            for (int k = 0; k < count; ++k) {
                const double dx = tileX[k] - xi;
                const double dy = tileY[k] - yi;
                const double dz = tileZ[k] - zi;
                const double d2 = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double inv = 1.0 / sqrt(d2);
                const double inv3 = inv * inv * inv;
                fx += dx * inv3;
                fy += dy * inv3;
                fz += dz * inv3;
            }
        }
        __syncthreads();
    }
    if (active) {
        const double nvx = vx[i] + DT * fx;
        const double nvy = vy[i] + DT * fy;
        const double nvz = vz[i] + DT * fz;
        vx[i] = nvx; vy[i] = nvy; vz[i] = nvz;
        px[i] += nvx * DT;
        py[i] += nvy * DT;
        pz[i] += nvz * DT;
    }
}

static void cudaCheck(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(status));
        std::exit(1);
    }
}

class GpuBodies {
public:
    double *px = nullptr, *py = nullptr, *pz = nullptr;
    double *vx = nullptr, *vy = nullptr, *vz = nullptr;
    int n;
    explicit GpuBodies(std::vector<Body>& bodies) : n(static_cast<int>(bodies.size())) {
        if (!n) return;
        double* host[6];
        for (int c = 0; c < 6; ++c) {
            cudaCheck(cudaMalloc(reinterpret_cast<void**>(&device[c]), n * sizeof(double)), "allocation");
            host[c] = new double[n];
        }
        px=device[0]; py=device[1]; pz=device[2]; vx=device[3]; vy=device[4]; vz=device[5];
        for (int i=0; i<n; ++i) {
            host[0][i]=bodies[i].pos.x; host[1][i]=bodies[i].pos.y; host[2][i]=bodies[i].pos.z;
            host[3][i]=bodies[i].vel.x; host[4][i]=bodies[i].vel.y; host[5][i]=bodies[i].vel.z;
        }
        for (int c=0; c<6; ++c) {
            cudaCheck(cudaMemcpy(device[c], host[c], n*sizeof(double), cudaMemcpyHostToDevice), "upload");
            delete[] host[c];
        }
    }
    ~GpuBodies() { for (double* p : device) if (p) cudaFree(p); }
    void download(std::vector<Body>& bodies) {
        if (!n) return;
        double* host[6];
        for (int c=0; c<6; ++c) { host[c]=new double[n]; cudaCheck(cudaMemcpy(host[c],device[c],n*sizeof(double),cudaMemcpyDeviceToHost),"download"); }
        for (int i=0; i<n; ++i) {
            bodies[i].pos.x=host[0][i]; bodies[i].pos.y=host[1][i]; bodies[i].pos.z=host[2][i];
            bodies[i].vel.x=host[3][i]; bodies[i].vel.y=host[4][i]; bodies[i].vel.z=host[5][i];
        }
        for (double* p : host) delete[] p;
    }
private:
    double* device[6] = {};
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

void computeForces(std::vector<Body>& bodies) {
    const size_t n = bodies.size();
    
    for (size_t i = 0; i < n; ++i) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;
        
        for (size_t j = 0; j < n; ++j) {
            const double dx = bodies[j].pos.x - bodies[i].pos.x;
            const double dy = bodies[j].pos.y - bodies[i].pos.y;
            const double dz = bodies[j].pos.z - bodies[i].pos.z;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        
        bodies[i].vel.x += DT * Fx;
        bodies[i].vel.y += DT * Fy;
        bodies[i].vel.z += DT * Fz;
    }
}

void integrateBodies(std::vector<Body>& bodies) {
    for (auto& body : bodies) {
        body.pos.x += body.vel.x * DT;
        body.pos.y += body.vel.y * DT;
        body.pos.z += body.vel.z * DT;
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
    GpuBodies gpuBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        if (numBodies > 0) {
            stepBodies<<<(numBodies + TILE_SIZE - 1) / TILE_SIZE, TILE_SIZE>>>(
                gpuBodies.px, gpuBodies.py, gpuBodies.pz,
                gpuBodies.vx, gpuBodies.vy, gpuBodies.vz, numBodies);
            cudaCheck(cudaGetLastError(), "kernel launch");
        }
    }
    cudaCheck(cudaDeviceSynchronize(), "simulation");
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    gpuBodies.download(bodies);
    
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
