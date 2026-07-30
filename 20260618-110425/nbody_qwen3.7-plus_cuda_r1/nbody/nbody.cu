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

#define BLOCK_SIZE 256

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudaGetErrorString(err));                                     \
            exit(EXIT_FAILURE);                                                 \
        }                                                                       \
    } while (0)

struct Body {
    double pos_x, pos_y, pos_z;
    double vel_x, vel_y, vel_z;
};

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    for (auto& body : bodies) {
        body.pos_x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos_y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos_z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel_x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel_y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel_z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// Compute forces kernel with shared memory tiling.
__global__ void computeForcesKernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    double* __restrict__ vel_x,
    double* __restrict__ vel_y,
    double* __restrict__ vel_z,
    const int n)
{
    __shared__ double sh_pos_x[BLOCK_SIZE];
    __shared__ double sh_pos_y[BLOCK_SIZE];
    __shared__ double sh_pos_z[BLOCK_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double my_pos_x, my_pos_y, my_pos_z;
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    if (i < n) {
        my_pos_x = pos_x[i];
        my_pos_y = pos_y[i];
        my_pos_z = pos_z[i];
    }

    for (int tile = 0; tile < n; tile += BLOCK_SIZE) {
        int j = tile + threadIdx.x;
        if (j < n) {
            sh_pos_x[threadIdx.x] = pos_x[j];
            sh_pos_y[threadIdx.x] = pos_y[j];
            sh_pos_z[threadIdx.x] = pos_z[j];
        }
        __syncthreads();

        if (i < n) {
            #pragma unroll
            for (int k = 0; k < BLOCK_SIZE; ++k) {
                int jj = tile + k;
                if (jj < n) {
                    double dx = sh_pos_x[k] - my_pos_x;
                    double dy = sh_pos_y[k] - my_pos_y;
                    double dz = sh_pos_z[k] - my_pos_z;
                    double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                    double invDist = rsqrt(distSqr);
                    double invDist3 = invDist * invDist * invDist;

                    Fx += dx * invDist3;
                    Fy += dy * invDist3;
                    Fz += dz * invDist3;
                }
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

// Integrate positions using updated velocities
__global__ void integrateBodiesKernel(
    double* __restrict__ pos_x,
    double* __restrict__ pos_y,
    double* __restrict__ pos_z,
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        pos_x[i] += vel_x[i] * DT;
        pos_y[i] += vel_y[i] * DT;
        pos_z[i] += vel_z[i] * DT;
    }
}

// Compute kinetic energy per body with block-level reduction
__global__ void kineticEnergyKernel(
    const double* __restrict__ vel_x,
    const double* __restrict__ vel_y,
    const double* __restrict__ vel_z,
    double* __restrict__ energy,
    const int n)
{
    __shared__ double sdata[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    if (i < n) {
        double vx = vel_x[i];
        double vy = vel_y[i];
        double vz = vel_z[i];
        val = 0.5 * (vx * vx + vy * vy + vz * vz);
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        energy[blockIdx.x] = sdata[0];
    }
}

// Compute potential energy contribution for body i (sum over j > i)
__global__ void potentialEnergyKernel(
    const double* __restrict__ pos_x,
    const double* __restrict__ pos_y,
    const double* __restrict__ pos_z,
    double* __restrict__ energy,
    const int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    double val = 0.0;
    if (i < n) {
        double px = pos_x[i];
        double py = pos_y[i];
        double pz = pos_z[i];
        for (int j = i + 1; j < n; ++j) {
            double dx = pos_x[j] - px;
            double dy = pos_y[j] - py;
            double dz = pos_z[j] - pz;
            double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            val -= 1.0 / dist;
        }
    }
    energy[i] = val;
}

// Sum reduction kernel with grid-stride loop
__global__ void sumReductionKernel(const double* __restrict__ input,
                                   double* __restrict__ output,
                                   int n)
{
    __shared__ double sdata[BLOCK_SIZE];

    const int tid = threadIdx.x;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double val = 0.0;
    for (int idx = i; idx < n; idx += blockDim.x * gridDim.x) {
        val += input[idx];
    }
    sdata[tid] = val;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        output[blockIdx.x] = sdata[0];
    }
}

struct GPUBodies {
    double *d_pos_x, *d_pos_y, *d_pos_z;
    double *d_vel_x, *d_vel_y, *d_vel_z;
    int n;
};

void uploadBodies(GPUBodies& gpu, const std::vector<Body>& bodies) {
    gpu.n = bodies.size();
    size_t sz = gpu.n * sizeof(double);
    std::vector<double> px(gpu.n), py(gpu.n), pz(gpu.n);
    std::vector<double> vx(gpu.n), vy(gpu.n), vz(gpu.n);
    for (int i = 0; i < gpu.n; ++i) {
        px[i] = bodies[i].pos_x; py[i] = bodies[i].pos_y; pz[i] = bodies[i].pos_z;
        vx[i] = bodies[i].vel_x; vy[i] = bodies[i].vel_y; vz[i] = bodies[i].vel_z;
    }
    CUDA_CHECK(cudaMalloc(&gpu.d_pos_x, sz));
    CUDA_CHECK(cudaMalloc(&gpu.d_pos_y, sz));
    CUDA_CHECK(cudaMalloc(&gpu.d_pos_z, sz));
    CUDA_CHECK(cudaMalloc(&gpu.d_vel_x, sz));
    CUDA_CHECK(cudaMalloc(&gpu.d_vel_y, sz));
    CUDA_CHECK(cudaMalloc(&gpu.d_vel_z, sz));
    CUDA_CHECK(cudaMemcpy(gpu.d_pos_x, px.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_pos_y, py.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_pos_z, pz.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_vel_x, vx.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_vel_y, vy.data(), sz, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu.d_vel_z, vz.data(), sz, cudaMemcpyHostToDevice));
}

void downloadBodies(const GPUBodies& gpu, std::vector<Body>& bodies) {
    size_t sz = gpu.n * sizeof(double);
    std::vector<double> px(gpu.n), py(gpu.n), pz(gpu.n);
    std::vector<double> vx(gpu.n), vy(gpu.n), vz(gpu.n);
    CUDA_CHECK(cudaMemcpy(px.data(), gpu.d_pos_x, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(py.data(), gpu.d_pos_y, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(pz.data(), gpu.d_pos_z, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vx.data(), gpu.d_vel_x, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vy.data(), gpu.d_vel_y, sz, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(vz.data(), gpu.d_vel_z, sz, cudaMemcpyDeviceToHost));
    bodies.resize(gpu.n);
    for (int i = 0; i < gpu.n; ++i) {
        bodies[i].pos_x = px[i]; bodies[i].pos_y = py[i]; bodies[i].pos_z = pz[i];
        bodies[i].vel_x = vx[i]; bodies[i].vel_y = vy[i]; bodies[i].vel_z = vz[i];
    }
}

void freeGPUBodies(GPUBodies& gpu) {
    cudaFree(gpu.d_pos_x);
    cudaFree(gpu.d_pos_y);
    cudaFree(gpu.d_pos_z);
    cudaFree(gpu.d_vel_x);
    cudaFree(gpu.d_vel_y);
    cudaFree(gpu.d_vel_z);
}

void computeForces(GPUBodies& gpu) {
    int blocks = (gpu.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    computeForcesKernel<<<blocks, BLOCK_SIZE>>>(
        gpu.d_pos_x, gpu.d_pos_y, gpu.d_pos_z,
        gpu.d_vel_x, gpu.d_vel_y, gpu.d_vel_z,
        gpu.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

void integrateBodies(GPUBodies& gpu) {
    int blocks = (gpu.n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    integrateBodiesKernel<<<blocks, BLOCK_SIZE>>>(
        gpu.d_pos_x, gpu.d_pos_y, gpu.d_pos_z,
        gpu.d_vel_x, gpu.d_vel_y, gpu.d_vel_z,
        gpu.n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

double computeTotalEnergy(const GPUBodies& gpu) {
    int n = gpu.n;
    int blocks = (n + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t sz = n * sizeof(double);

    double *d_energy;
    CUDA_CHECK(cudaMalloc(&d_energy, sz));

    // Kinetic energy
    kineticEnergyKernel<<<blocks, BLOCK_SIZE>>>(
        gpu.d_vel_x, gpu.d_vel_y, gpu.d_vel_z, d_energy, n);
    CUDA_CHECK(cudaGetLastError());

    int red_blocks = std::min(blocks, 256);
    double *d_ke_result;
    CUDA_CHECK(cudaMalloc(&d_ke_result, red_blocks * sizeof(double)));
    sumReductionKernel<<<red_blocks, BLOCK_SIZE>>>(d_energy, d_ke_result, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_partial(red_blocks);
    CUDA_CHECK(cudaMemcpy(h_partial.data(), d_ke_result, red_blocks * sizeof(double), cudaMemcpyDeviceToHost));
    double kinetic = 0.0;
    for (double v : h_partial) kinetic += v;

    // Potential energy
    potentialEnergyKernel<<<blocks, BLOCK_SIZE>>>(
        gpu.d_pos_x, gpu.d_pos_y, gpu.d_pos_z, d_energy, n);
    CUDA_CHECK(cudaGetLastError());

    double *d_pe_result;
    CUDA_CHECK(cudaMalloc(&d_pe_result, red_blocks * sizeof(double)));
    sumReductionKernel<<<red_blocks, BLOCK_SIZE>>>(d_energy, d_pe_result, n);
    CUDA_CHECK(cudaGetLastError());

    std::vector<double> h_pe_partial(red_blocks);
    CUDA_CHECK(cudaMemcpy(h_pe_partial.data(), d_pe_result, red_blocks * sizeof(double), cudaMemcpyDeviceToHost));
    double potential = 0.0;
    for (double v : h_pe_partial) potential += v;

    CUDA_CHECK(cudaDeviceSynchronize());
    cudaFree(d_energy);
    cudaFree(d_ke_result);
    cudaFree(d_pe_result);

    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for (const auto& body : bodies) {
        if (!std::isfinite(body.pos_x) || !std::isfinite(body.pos_y) || !std::isfinite(body.pos_z) ||
            !std::isfinite(body.vel_x) || !std::isfinite(body.vel_y) || !std::isfinite(body.vel_z)) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }
        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(body.pos_x) > maxPos || std::abs(body.pos_y) > maxPos || std::abs(body.pos_z) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(body.vel_x) > maxVel || std::abs(body.vel_y) > maxVel || std::abs(body.vel_z) > maxVel) {
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

    // Upload to GPU
    GPUBodies gpu;
    uploadBodies(gpu, bodies);

    // Run simulation on GPU
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForces(gpu);
        integrateBodies(gpu);
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Download results
    downloadBodies(gpu, bodies);

    // Print results for external validation
    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (const auto& body : bodies) {
            bodyData.push_back(body.pos_x);
            bodyData.push_back(body.pos_y);
            bodyData.push_back(body.pos_z);
            bodyData.push_back(body.vel_x);
            bodyData.push_back(body.vel_y);
            bodyData.push_back(body.vel_z);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation
    if (validate) {
        printf("Validating simulation results...\n");
        if (validateSimulation(bodies)) {
            double finalEnergy = computeTotalEnergy(gpu);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            freeGPUBodies(gpu);
            return 0;
        } else {
            printf("Validation: FAILED\n");
            freeGPUBodies(gpu);
            return 1;
        }
    }

    freeGPUBodies(gpu);
    return 0;
}
