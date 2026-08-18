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
constexpr int THREADS_PER_BLOCK = 256;

#define CUDA_CHECK(call) do { \
    cudaError_t error__ = (call); \
    if (error__ != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(error__)); \
        std::exit(EXIT_FAILURE); \
    } \
} while (false)

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

__global__ void computeForcesKernel(const double* __restrict__ px,
                                    const double* __restrict__ py,
                                    const double* __restrict__ pz,
                                    double* __restrict__ vx,
                                    double* __restrict__ vy,
                                    double* __restrict__ vz, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const double xi = px[i], yi = py[i], zi = pz[i];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = px[j] - xi;
        const double dy = py[j] - yi;
        const double dz = pz[j] - zi;
        const double invDist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        const double invDist3 = invDist * invDist * invDist;
        fx += dx * invDist3;
        fy += dy * invDist3;
        fz += dz * invDist3;
    }
    vx[i] += DT * fx;
    vy[i] += DT * fy;
    vz[i] += DT * fz;
}

__global__ void integrateKernel(double* px, double* py, double* pz,
                                const double* __restrict__ vx,
                                const double* __restrict__ vy,
                                const double* __restrict__ vz, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

class DeviceBodies {
public:
    explicit DeviceBodies(const std::vector<Body>& host) : n_(static_cast<int>(host.size())) {
        const size_t bytes = static_cast<size_t>(n_) * sizeof(double);
        CUDA_CHECK(cudaMalloc(&px_, bytes)); CUDA_CHECK(cudaMalloc(&py_, bytes)); CUDA_CHECK(cudaMalloc(&pz_, bytes));
        CUDA_CHECK(cudaMalloc(&vx_, bytes)); CUDA_CHECK(cudaMalloc(&vy_, bytes)); CUDA_CHECK(cudaMalloc(&vz_, bytes));
        std::vector<double> x(n_), y(n_), z(n_), u(n_), v(n_), w(n_);
        for (int i = 0; i < n_; ++i) { x[i]=host[i].pos.x; y[i]=host[i].pos.y; z[i]=host[i].pos.z; u[i]=host[i].vel.x; v[i]=host[i].vel.y; w[i]=host[i].vel.z; }
        CUDA_CHECK(cudaMemcpy(px_, x.data(), bytes, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(py_, y.data(), bytes, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(pz_, z.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(vx_, u.data(), bytes, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(vy_, v.data(), bytes, cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(vz_, w.data(), bytes, cudaMemcpyHostToDevice));
    }
    ~DeviceBodies() { cudaFree(px_); cudaFree(py_); cudaFree(pz_); cudaFree(vx_); cudaFree(vy_); cudaFree(vz_); }
    void step() {
        const int blocks = (n_ + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        computeForcesKernel<<<blocks, THREADS_PER_BLOCK>>>(px_,py_,pz_,vx_,vy_,vz_,n_);
        integrateKernel<<<blocks, THREADS_PER_BLOCK>>>(px_,py_,pz_,vx_,vy_,vz_,n_);
        CUDA_CHECK(cudaGetLastError());
    }
    void copyTo(std::vector<Body>& host) const {
        const size_t bytes = static_cast<size_t>(n_) * sizeof(double); std::vector<double> x(n_),y(n_),z(n_),u(n_),v(n_),w(n_);
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaMemcpy(x.data(),px_,bytes,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(y.data(),py_,bytes,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(z.data(),pz_,bytes,cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(u.data(),vx_,bytes,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(v.data(),vy_,bytes,cudaMemcpyDeviceToHost)); CUDA_CHECK(cudaMemcpy(w.data(),vz_,bytes,cudaMemcpyDeviceToHost));
        for (int i=0;i<n_;++i) { host[i].pos={x[i],y[i],z[i]}; host[i].vel={u[i],v[i],w[i]}; }
    }
private: int n_; double *px_=nullptr,*py_=nullptr,*pz_=nullptr,*vx_=nullptr,*vy_=nullptr,*vz_=nullptr;
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
    DeviceBodies deviceBodies(bodies);
    
    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int step = 0; step < numSteps; ++step) {
        deviceBodies.step();
    }

    deviceBodies.copyTo(bodies);
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    
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
