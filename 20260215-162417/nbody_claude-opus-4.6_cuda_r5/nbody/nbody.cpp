#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

#define SOFTENING 1e-9
#define DT 0.01
#define BLOCK_SIZE 256

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
} while(0)

__global__ void computeForcesKernel(const double* __restrict__ px,
                                     const double* __restrict__ py,
                                     const double* __restrict__ pz,
                                     double* __restrict__ vx,
                                     double* __restrict__ vy,
                                     double* __restrict__ vz,
                                     int n) {
    extern __shared__ double smem[];
    double* spx = smem;
    double* spy = spx + blockDim.x;
    double* spz = spy + blockDim.x;

    int i = blockIdx.x * blockDim.x + threadIdx.x;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    double myx = 0.0, myy = 0.0, myz = 0.0;

    if (i < n) {
        myx = px[i];
        myy = py[i];
        myz = pz[i];
    }

    int numTiles = (n + (int)blockDim.x - 1) / (int)blockDim.x;

    for (int tile = 0; tile < numTiles; ++tile) {
        int idx = tile * blockDim.x + threadIdx.x;
        if (idx < n) {
            spx[threadIdx.x] = px[idx];
            spy[threadIdx.x] = py[idx];
            spz[threadIdx.x] = pz[idx];
        }
        __syncthreads();

        if (i < n) {
            int tileEnd = min((int)blockDim.x, n - tile * (int)blockDim.x);
            for (int k = 0; k < tileEnd; ++k) {
                double dx = spx[k] - myx;
                double dy = spy[k] - myy;
                double dz = spz[k] - myz;
                double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                double invDist = 1.0 / sqrt(distSqr);
                double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
    }
}

__global__ void integrateBodiesKernel(double* __restrict__ px,
                                       double* __restrict__ py,
                                       double* __restrict__ pz,
                                       const double* __restrict__ vx,
                                       const double* __restrict__ vy,
                                       const double* __restrict__ vz,
                                       int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        px[i] += vx[i] * DT;
        py[i] += vy[i] * DT;
        pz[i] += vz[i] * DT;
    }
}

double computeTotalEnergy(const double* px, const double* py, const double* pz,
                          const double* vx, const double* vy, const double* vz, int n) {
    double energy = 0.0;

    for (int i = 0; i < n; ++i) {
        energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    for (int i = 0; i < n; ++i) {
        for (int j = i + 1; j < n; ++j) {
            double dx = px[j] - px[i];
            double dy = py[j] - py[i];
            double dz = pz[j] - pz[i];
            double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

bool validateSimulation(const double* px, const double* py, const double* pz,
                        const double* vx, const double* vy, const double* vz, int n) {
    for (int i = 0; i < n; ++i) {
        if (!std::isfinite(px[i]) || !std::isfinite(py[i]) || !std::isfinite(pz[i]) ||
            !std::isfinite(vx[i]) || !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(px[i]) > maxPos || std::abs(py[i]) > maxPos || std::abs(pz[i]) > maxPos) {
            printf("Validation failed: body position exceeds reasonable bounds\n");
            return false;
        }
        if (std::abs(vx[i]) > maxVel || std::abs(vy[i]) > maxVel || std::abs(vz[i]) > maxVel) {
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

    // Host SoA arrays
    std::vector<double> h_px(numBodies), h_py(numBodies), h_pz(numBodies);
    std::vector<double> h_vx(numBodies), h_vy(numBodies), h_vz(numBodies);

    // Initialize bodies (same rand_r call order as original)
    unsigned int seed = 42;
    for (int i = 0; i < numBodies; ++i) {
        h_px[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        h_py[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        h_pz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        h_vx[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        h_vy[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        h_vz[i] = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }

    // Device arrays
    double *d_px, *d_py, *d_pz, *d_vx, *d_vy, *d_vz;
    size_t bytes = numBodies * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_px, bytes));
    CUDA_CHECK(cudaMalloc(&d_py, bytes));
    CUDA_CHECK(cudaMalloc(&d_pz, bytes));
    CUDA_CHECK(cudaMalloc(&d_vx, bytes));
    CUDA_CHECK(cudaMalloc(&d_vy, bytes));
    CUDA_CHECK(cudaMalloc(&d_vz, bytes));

    CUDA_CHECK(cudaMemcpy(d_px, h_px.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_py, h_py.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pz, h_pz.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vz, h_vz.data(), bytes, cudaMemcpyHostToDevice));

    int blocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
    size_t sharedBytes = 3 * BLOCK_SIZE * sizeof(double);

    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        computeForcesKernel<<<blocks, BLOCK_SIZE, sharedBytes>>>(
            d_px, d_py, d_pz, d_vx, d_vy, d_vz, numBodies);
        integrateBodiesKernel<<<blocks, BLOCK_SIZE>>>(
            d_px, d_py, d_pz, d_vx, d_vy, d_vz, numBodies);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Copy results back
    CUDA_CHECK(cudaMemcpy(h_px.data(), d_px, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_py.data(), d_py, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_pz.data(), d_pz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost));

    if (printResults) {
        std::vector<double> bodyData;
        bodyData.reserve(numBodies * 6);
        for (int i = 0; i < numBodies; ++i) {
            bodyData.push_back(h_px[i]);
            bodyData.push_back(h_py[i]);
            bodyData.push_back(h_pz[i]);
            bodyData.push_back(h_vx[i]);
            bodyData.push_back(h_vy[i]);
            bodyData.push_back(h_vz[i]);
        }
        print_results(bodyData, "Bodies");
    }

    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(h_px.data(), h_py.data(), h_pz.data(),
                               h_vx.data(), h_vy.data(), h_vz.data(), numBodies)) {
            double finalEnergy = computeTotalEnergy(h_px.data(), h_py.data(), h_pz.data(),
                                                     h_vx.data(), h_vy.data(), h_vz.data(), numBodies);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
            CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
            CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
            return 0;
        } else {
            printf("Validation: FAILED\n");
            CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
            CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));
            return 1;
        }
    }

    CUDA_CHECK(cudaFree(d_px)); CUDA_CHECK(cudaFree(d_py)); CUDA_CHECK(cudaFree(d_pz));
    CUDA_CHECK(cudaFree(d_vx)); CUDA_CHECK(cudaFree(d_vy)); CUDA_CHECK(cudaFree(d_vz));

    return 0;
}
