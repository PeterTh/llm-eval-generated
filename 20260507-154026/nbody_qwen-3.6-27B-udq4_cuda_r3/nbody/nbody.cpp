#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

// SOA layout for coalesced memory access on GPU
struct Bodies {
    double* px;
    double* py;
    double* pz;
    double* vx;
    double* vy;
    double* vz;
    size_t n;
};

// Initialize bodies on host with the same PRNG as the original code
static void initBodiesOnHost(size_t n, double* px, double* py, double* pz,
                             double* vx, double* vy, double* vz, unsigned int seed) {
    for (size_t i = 0; i < n; ++i) {
        px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

// CUDA kernel: compute gravitational forces (N-body, O(N^2))
// Each thread handles one body i, summing forces from all bodies j.
// Uses shared-memory tiling for efficient memory access.
__global__ void computeForcesKernel(const double* __restrict__ px,
                                    const double* __restrict__ py,
                                    const double* __restrict__ pz,
                                    double* __restrict__ vx,
                                    double* __restrict__ vy,
                                    double* __restrict__ vz,
                                    const size_t n) {
    extern __shared__ double shared[];

    double* s_px = shared;
    double* s_py = shared + blockDim.x;
    double* s_pz = shared + 2 * blockDim.x;

    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        double Fx = 0.0, Fy = 0.0, Fz = 0.0;

        const double pi_x = px[i];
        const double pi_y = py[i];
        const double pi_z = pz[i];

        // Iterate over all bodies in tiles
        for (size_t base = 0; base < n; base += blockDim.x) {
            // Load tile into shared memory
            const size_t j = base + threadIdx.x;
            if (j < n) {
                s_px[threadIdx.x] = px[j];
                s_py[threadIdx.x] = py[j];
                s_pz[threadIdx.x] = pz[j];
            }
            __syncthreads();

            // Compute forces from this tile
            const size_t tile_end = min(base + blockDim.x, n);
            for (size_t k = base; k < tile_end; ++k) {
                const double dx = s_px[k - base] - pi_x;
                const double dy = s_py[k - base] - pi_y;
                const double dz = s_pz[k - base] - pi_z;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

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
}

// CUDA kernel: integrate positions from velocities
__global__ void integrateKernel(double* __restrict__ px,
                                double* __restrict__ py,
                                double* __restrict__ pz,
                                const double* __restrict__ vx,
                                const double* __restrict__ vy,
                                const double* __restrict__ vz,
                                const size_t n) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

// CUDA kernel: compute total energy (kinetic + potential)
// Uses atomicAdd for the reduction.
__global__ void computeEnergyKernel(const double* __restrict__ px,
                                    const double* __restrict__ py,
                                    const double* __restrict__ pz,
                                    const double* __restrict__ vx,
                                    const double* __restrict__ vy,
                                    const double* __restrict__ vz,
                                    const size_t n,
                                    double* energy) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    double local_energy = 0.0;

    // Kinetic energy: each thread handles some bodies
    for (size_t i = tid; i < n; i += stride) {
        local_energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    // Potential energy: each thread handles some pairs (i, j) with i < j
    for (size_t i = tid; i < n; i += stride) {
        const double pi_x = px[i];
        const double pi_y = py[i];
        const double pi_z = pz[i];
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = px[j] - pi_x;
            const double dy = py[j] - pi_y;
            const double dz = pz[j] - pi_z;
            const double dist = sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            local_energy -= 1.0 / dist;
        }
    }

    atomicAdd(energy, local_energy);
}

// Validation kernel: check for NaN/Inf and extreme values
__global__ void validateKernel(const double* __restrict__ px,
                               const double* __restrict__ py,
                               const double* __restrict__ pz,
                               const double* __restrict__ vx,
                               const double* __restrict__ vy,
                               const double* __restrict__ vz,
                               const size_t n,
                               int* has_error) {
    const size_t tid = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const size_t stride = static_cast<size_t>(blockDim.x) * gridDim.x;

    for (size_t i = tid; i < n; i += stride) {
        const double maxPos = 1e6;
        const double maxVel = 1e6;

        if (!isfinite(px[i]) || !isfinite(py[i]) || !isfinite(pz[i]) ||
            !isfinite(vx[i]) || !isfinite(vy[i]) || !isfinite(vz[i])) {
            atomicOr(has_error, 1);
            return;
        }
        if (fabs(px[i]) > maxPos || fabs(py[i]) > maxPos || fabs(pz[i]) > maxPos) {
            atomicOr(has_error, 2);
            return;
        }
        if (fabs(vx[i]) > maxVel || fabs(vy[i]) > maxVel || fabs(vz[i]) > maxVel) {
            atomicOr(has_error, 3);
            return;
        }
    }
}

static void checkCuda(cudaError_t err, const char* msg) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s - %s\n", msg, cudaGetErrorString(err));
        exit(1);
    }
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

    const size_t n = static_cast<size_t>(numBodies);

    // Allocate device memory (SOA layout for coalesced access)
    double *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
    const size_t bytes = n * sizeof(double);
    checkCuda(cudaMalloc(&d_px, bytes), "alloc px");
    checkCuda(cudaMalloc(&d_py, bytes), "alloc py");
    checkCuda(cudaMalloc(&d_pz, bytes), "alloc pz");
    checkCuda(cudaMalloc(&d_vx, bytes), "alloc vx");
    checkCuda(cudaMalloc(&d_vy, bytes), "alloc vy");
    checkCuda(cudaMalloc(&d_vz, bytes), "alloc vz");

    // Initialize bodies on host with same PRNG sequence
    std::vector<double> h_px(n), h_py(n), h_pz(n);
    std::vector<double> h_vx(n), h_vy(n), h_vz(n);
    unsigned int seed = 42;
    initBodiesOnHost(n, h_px.data(), h_py.data(), h_pz.data(),
                     h_vx.data(), h_vy.data(), h_vz.data(), seed);

    // Copy to device
    checkCuda(cudaMemcpy(d_px, h_px.data(), bytes, cudaMemcpyHostToDevice), "copy px");
    checkCuda(cudaMemcpy(d_py, h_py.data(), bytes, cudaMemcpyHostToDevice), "copy py");
    checkCuda(cudaMemcpy(d_pz, h_pz.data(), bytes, cudaMemcpyHostToDevice), "copy pz");
    checkCuda(cudaMemcpy(d_vx, h_vx.data(), bytes, cudaMemcpyHostToDevice), "copy vx");
    checkCuda(cudaMemcpy(d_vy, h_vy.data(), bytes, cudaMemcpyHostToDevice), "copy vy");
    checkCuda(cudaMemcpy(d_vz, h_vz.data(), bytes, cudaMemcpyHostToDevice), "copy vz");

    // Launch configuration
    constexpr int BLOCK_SIZE = 256;
    const int numBlocks = min(static_cast<int>((n + BLOCK_SIZE - 1) / BLOCK_SIZE), 256);
    // Shared memory: 3 arrays of BLOCK_SIZE doubles
    const size_t sharedMemBytes = 3 * BLOCK_SIZE * sizeof(double);

    // Warmup + timing
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<numBlocks, BLOCK_SIZE, sharedMemBytes>>>(
            d_px, d_py, d_pz, d_vx, d_vy, d_vz, n);
        checkCuda(cudaGetLastError(), "computeForcesKernel");

        integrateKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_pz, d_vx, d_vy, d_vz, n);
        checkCuda(cudaGetLastError(), "integrateKernel");
    }

    // Synchronize to ensure all GPU work is done before timing
    checkCuda(cudaDeviceSynchronize(), "device synchronize");

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        // Copy back to host
        checkCuda(cudaMemcpy(h_px.data(), d_px, bytes, cudaMemcpyDeviceToHost), "copy px back");
        checkCuda(cudaMemcpy(h_py.data(), d_py, bytes, cudaMemcpyDeviceToHost), "copy py back");
        checkCuda(cudaMemcpy(h_pz.data(), d_pz, bytes, cudaMemcpyDeviceToHost), "copy pz back");
        checkCuda(cudaMemcpy(h_vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost), "copy vx back");
        checkCuda(cudaMemcpy(h_vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost), "copy vy back");
        checkCuda(cudaMemcpy(h_vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost), "copy vz back");

        std::vector<double> bodyData;
        bodyData.reserve(n * 6);
        for (size_t i = 0; i < n; ++i) {
            bodyData.push_back(h_px[i]);
            bodyData.push_back(h_py[i]);
            bodyData.push_back(h_pz[i]);
            bodyData.push_back(h_vx[i]);
            bodyData.push_back(h_vy[i]);
            bodyData.push_back(h_vz[i]);
        }
        print_results(bodyData, "Bodies");
    }

    // Validation
    if (validate) {
        printf("Validating simulation results...\n");

        int h_error = 0;
        int *d_error;
        checkCuda(cudaMalloc(&d_error, sizeof(int)), "alloc error flag");
        checkCuda(cudaMemset(d_error, 0, sizeof(int)), "memset error flag");

        validateKernel<<<numBlocks, BLOCK_SIZE>>>(
            d_px, d_py, d_pz, d_vx, d_vy, d_vz, n, d_error);
        checkCuda(cudaGetLastError(), "validateKernel");
        checkCuda(cudaDeviceSynchronize(), "device synchronize");

        checkCuda(cudaMemcpy(&h_error, d_error, sizeof(int), cudaMemcpyDeviceToHost), "copy error");
        cudaFree(d_error);

        if (h_error == 0) {
            // Compute final energy on GPU
            double h_energy = 0.0;
            double *d_energy;
            checkCuda(cudaMalloc(&d_energy, sizeof(double)), "alloc energy");
            checkCuda(cudaMemset(d_energy, 0, sizeof(double)), "memset energy");

            computeEnergyKernel<<<numBlocks, BLOCK_SIZE>>>(
                d_px, d_py, d_pz, d_vx, d_vy, d_vz, n, d_energy);
            checkCuda(cudaGetLastError(), "computeEnergyKernel");
            checkCuda(cudaDeviceSynchronize(), "device synchronize");

            checkCuda(cudaMemcpy(&h_energy, d_energy, sizeof(double), cudaMemcpyDeviceToHost), "copy energy");
            cudaFree(d_energy);

            printf("Final energy: %.6f\n", h_energy);
            printf("Validation: PASSED\n");

            // Cleanup
            cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            return 0;
        } else {
            if (h_error == 1) printf("Validation failed: found NaN or Inf value in body state\n");
            else if (h_error == 2) printf("Validation failed: body position exceeds reasonable bounds\n");
            else if (h_error == 3) printf("Validation failed: body velocity exceeds reasonable bounds\n");
            printf("Validation: FAILED\n");

            // Cleanup
            cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
            cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);
            return 1;
        }
    }

    // Cleanup
    cudaFree(d_px); cudaFree(d_py); cudaFree(d_pz);
    cudaFree(d_vx); cudaFree(d_vy); cudaFree(d_vz);

    return 0;
}
