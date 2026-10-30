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

// Device arrays
static double *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
static size_t d_n = 0;

__global__ void computeForcesKernel(size_t n, double* px, double* py, double* pz,
                                    double* vx, double* vy, double* vz) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double xi = px[i], yi = py[i], zi = pz[i];
    for (size_t j = 0; j < n; ++j) {
        const double dx = px[j] - xi;
        const double dy = py[j] - yi;
        const double dz = pz[j] - zi;
        const double distSqr = dx*dx + dy*dy + dz*dz + SOFTENING;
        const double invDist = 1.0 / sqrt(distSqr);
        const double invDist3 = invDist * invDist * invDist;
        Fx += dx * invDist3;
        Fy += dy * invDist3;
        Fz += dz * invDist3;
    }
    vx[i] += DT * Fx;
    vy[i] += DT * Fy;
    vz[i] += DT * Fz;
}

__global__ void integrateKernel(size_t n, double* px, double* py, double* pz,
                               const double* vx, const double* vy, const double* vz) {
    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    px[i] += vx[i] * DT;
    py[i] += vy[i] * DT;
    pz[i] += vz[i] * DT;
}

void ensureDevice(size_t n) {
    if (d_n == n) return;
    if (d_n > 0) {
        cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
        cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
    }
    cudaMalloc(&d_px, n*sizeof(double));
    cudaMalloc(&d_py, n*sizeof(double));
    cudaMalloc(&d_pz, n*sizeof(double));
    cudaMalloc(&d_vx, n*sizeof(double));
    cudaMalloc(&d_vy, n*sizeof(double));
    cudaMalloc(&d_vz, n*sizeof(double));
    d_n = n;
}

void upload(const std::vector<Body>& bodies) {
    size_t n = bodies.size();
    ensureDevice(n);
    std::vector<double> px(n),py(n),pz(n),vx(n),vy(n),vz(n);
    for (size_t i=0;i<n;++i){px[i]=bodies[i].pos.x;py[i]=bodies[i].pos.y;pz[i]=bodies[i].pos.z;vx[i]=bodies[i].vel.x;vy[i]=bodies[i].vel.y;vz[i]=bodies[i].vel.z;}
    cudaMemcpy(d_px,px.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_py,py.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_pz,pz.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vx,vx.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vy,vy.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(d_vz,vz.data(),n*sizeof(double),cudaMemcpyHostToDevice);
}

void download(std::vector<Body>& bodies) {
    size_t n = bodies.size();
    std::vector<double> px(n),py(n),pz(n),vx(n),vy(n),vz(n);
    cudaMemcpy(px.data(),d_px,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(py.data(),d_py,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(pz.data(),d_pz,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(vx.data(),d_vx,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(vy.data(),d_vy,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(vz.data(),d_vz,n*sizeof(double),cudaMemcpyDeviceToHost);
    for (size_t i=0;i<n;++i){bodies[i].pos.x=px[i];bodies[i].pos.y=py[i];bodies[i].pos.z=pz[i];bodies[i].vel.x=vx[i];bodies[i].vel.y=vy[i];bodies[i].vel.z=vz[i];}
}

void computeForces(std::vector<Body>& bodies) {
    upload(bodies);
    int block = 256;
    int grid = (bodies.size() + block - 1)/block;
    computeForcesKernel<<<grid,block>>>(bodies.size(),d_px,d_py,d_pz,d_vx,d_vy,d_vz);
    cudaDeviceSynchronize();
    // keep device state; download velocities? integrate on device needs device vel already updated. But host bodies stale. We'll download at end? Need integrate on device.
    // integrate on device
    integrateKernel<<<grid,block>>>(bodies.size(),d_px,d_py,d_pz,d_vx,d_vy,d_vz);
    cudaDeviceSynchronize();
    download(bodies);
}

void integrateBodies(std::vector<Body>& bodies) {
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

    for (int step = 0; step < numSteps; ++step) {
        computeForces(bodies);
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
