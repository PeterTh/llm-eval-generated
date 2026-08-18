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

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
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

[[noreturn]] void cudaFailure(const char* operation, cudaError_t error) {
    fprintf(stderr, "CUDA error during %s: %s\n", operation, cudaGetErrorString(error));
    std::exit(EXIT_FAILURE);
}

void checkCuda(cudaError_t error, const char* operation) {
    if (error != cudaSuccess) cudaFailure(operation, error);
}

__global__ void computeForcesKernel(double* px, double* py, double* pz,
                                    double* vx, double* vy, double* vz, int n) {
    extern __shared__ double tile[];
    double* tx = tile;
    double* ty = tx + blockDim.x;
    double* tz = ty + blockDim.x;
    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + tid;
    const double ix = i < n ? px[i] : 0.0;
    const double iy = i < n ? py[i] : 0.0;
    const double iz = i < n ? pz[i] : 0.0;
    double fx = 0.0, fy = 0.0, fz = 0.0;

    for (int base = 0; base < n; base += blockDim.x) {
        const int j = base + tid;
        tx[tid] = j < n ? px[j] : 0.0;
        ty[tid] = j < n ? py[j] : 0.0;
        tz[tid] = j < n ? pz[j] : 0.0;
        __syncthreads();
        const int count = min(blockDim.x, n - base);
        for (int k = 0; k < count; ++k) {
            const double dx = tx[k] - ix;
            const double dy = ty[k] - iy;
            const double dz = tz[k] - iz;
            const double invDist = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            const double invDist3 = invDist * invDist * invDist;
            fx += dx * invDist3;
            fy += dy * invDist3;
            fz += dz * invDist3;
        }
        __syncthreads();
    }
    if (i < n) {
        vx[i] += DT * fx;
        vy[i] += DT * fy;
        vz[i] += DT * fz;
    }
}

__global__ void integrateBodiesKernel(double* px, double* py, double* pz,
                                      const double* vx, const double* vy,
                                      const double* vz, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

class GpuBodies {
public:
    explicit GpuBodies(const std::vector<Body>& host) : n_(static_cast<int>(host.size())) {
        const size_t bytes = host.size() * sizeof(double);
        std::vector<double> px(n_), py(n_), pz(n_), vx(n_), vy(n_), vz(n_);
        for (int i = 0; i < n_; ++i) {
            px[i] = host[i].pos.x; py[i] = host[i].pos.y; pz[i] = host[i].pos.z;
            vx[i] = host[i].vel.x; vy[i] = host[i].vel.y; vz[i] = host[i].vel.z;
        }
        checkCuda(cudaMalloc(&px_, bytes), "cudaMalloc"); checkCuda(cudaMalloc(&py_, bytes), "cudaMalloc");
        checkCuda(cudaMalloc(&pz_, bytes), "cudaMalloc"); checkCuda(cudaMalloc(&vx_, bytes), "cudaMalloc");
        checkCuda(cudaMalloc(&vy_, bytes), "cudaMalloc"); checkCuda(cudaMalloc(&vz_, bytes), "cudaMalloc");
        checkCuda(cudaMemcpy(px_, px.data(), bytes, cudaMemcpyHostToDevice), "upload positions");
        checkCuda(cudaMemcpy(py_, py.data(), bytes, cudaMemcpyHostToDevice), "upload positions");
        checkCuda(cudaMemcpy(pz_, pz.data(), bytes, cudaMemcpyHostToDevice), "upload positions");
        checkCuda(cudaMemcpy(vx_, vx.data(), bytes, cudaMemcpyHostToDevice), "upload velocities");
        checkCuda(cudaMemcpy(vy_, vy.data(), bytes, cudaMemcpyHostToDevice), "upload velocities");
        checkCuda(cudaMemcpy(vz_, vz.data(), bytes, cudaMemcpyHostToDevice), "upload velocities");
    }
    ~GpuBodies() { cudaFree(px_); cudaFree(py_); cudaFree(pz_); cudaFree(vx_); cudaFree(vy_); cudaFree(vz_); }
    void step() {
        constexpr int threads = 256;
        const int blocks = (n_ + threads - 1) / threads;
        computeForcesKernel<<<blocks, threads, 3 * threads * sizeof(double)>>>(px_, py_, pz_, vx_, vy_, vz_, n_);
        checkCuda(cudaGetLastError(), "force kernel");
        integrateBodiesKernel<<<blocks, threads>>>(px_, py_, pz_, vx_, vy_, vz_, n_);
        checkCuda(cudaGetLastError(), "integration kernel");
    }
    void download(std::vector<Body>& host) const {
        std::vector<double> px(n_), py(n_), pz(n_), vx(n_), vy(n_), vz(n_);
        const size_t bytes = n_ * sizeof(double);
        checkCuda(cudaMemcpy(px.data(), px_, bytes, cudaMemcpyDeviceToHost), "download positions");
        checkCuda(cudaMemcpy(py.data(), py_, bytes, cudaMemcpyDeviceToHost), "download positions");
        checkCuda(cudaMemcpy(pz.data(), pz_, bytes, cudaMemcpyDeviceToHost), "download positions");
        checkCuda(cudaMemcpy(vx.data(), vx_, bytes, cudaMemcpyDeviceToHost), "download velocities");
        checkCuda(cudaMemcpy(vy.data(), vy_, bytes, cudaMemcpyDeviceToHost), "download velocities");
        checkCuda(cudaMemcpy(vz.data(), vz_, bytes, cudaMemcpyDeviceToHost), "download velocities");
        for (int i = 0; i < n_; ++i) { host[i].pos = {px[i], py[i], pz[i]}; host[i].vel = {vx[i], vy[i], vz[i]}; }
    }
private:
    int n_; double *px_{}, *py_{}, *pz_{}, *vx_{}, *vy_{}, *vz_{};
};

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
        gpuBodies.step();
    }
    checkCuda(cudaDeviceSynchronize(), "simulation");
    gpuBodies.download(bodies);
    
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
