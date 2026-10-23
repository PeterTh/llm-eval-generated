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

constexpr int BLOCK_SIZE = 32;
constexpr int TILE_SIZE = 256;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(const double x = 0, const double y = 0, const double z = 0) noexcept : x(x), y(y), z(z) {}
};

struct Body {
    Vec3 pos;
    Vec3 vel;
};

#define CUDA_CHECK(call)                                                                         \
    do {                                                                                         \
        cudaError_t err_ = (call);                                                               \
        if (err_ != cudaSuccess) {                                                               \
            fprintf(stderr, "CUDA error (%s) at %s:%d\n", cudaGetErrorString(err_), __FILE__,    \
                    __LINE__);                                                                   \
            std::exit(1);                                                                        \
        }                                                                                        \
    } while (0)

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

// One thread per body. Each block streams every position through shared memory
// in tiles, so each position is fetched from the read-only cache once per
// block and tile. Positions are written to a separate buffer, which allows the
// velocity and position updates to be fused into a single kernel launch while
// every force still sees the positions from the beginning of the step.
__global__ void advanceBodiesKernel(const double* __restrict__ px, const double* __restrict__ py,
                                    const double* __restrict__ pz, double* __restrict__ vx,
                                    double* __restrict__ vy, double* __restrict__ vz,
                                    double* __restrict__ npx, double* __restrict__ npy,
                                    double* __restrict__ npz, int n) {
    __shared__ double sx[TILE_SIZE];
    __shared__ double sy[TILE_SIZE];
    __shared__ double sz[TILE_SIZE];

    const int i = blockIdx.x * blockDim.x + threadIdx.x;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) {
        xi = __ldg(px + i);
        yi = __ldg(py + i);
        zi = __ldg(pz + i);
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += TILE_SIZE) {
        for (int k = threadIdx.x; k < TILE_SIZE; k += blockDim.x) {
            const int t = base + k;
            if (t < n) {
                sx[k] = __ldg(px + t);
                sy[k] = __ldg(py + t);
                sz[k] = __ldg(pz + t);
            }
        }
        __syncthreads();

        const int jmax = min(TILE_SIZE, n - base);
#pragma unroll 4
        for (int j = 0; j < jmax; ++j) {
            const double dx = sx[j] - xi;
            const double dy = sy[j] - yi;
            const double dz = sz[j] - zi;
            // Keep the exact operation order and rounding of the original
            // implementation: dy*dy is rounded first and the remaining
            // products are fused, as emitted by the host compiler.
            const double distSqr = __fma_rn(dz, dz, __fma_rn(dx, dx, dy * dy)) + SOFTENING;
            const double invDist = 1.0 / std::sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;

            Fx = __fma_rn(dx, invDist3, Fx);
            Fy = __fma_rn(dy, invDist3, Fy);
            Fz = __fma_rn(dz, invDist3, Fz);
        }
        __syncthreads();
    }

    if (i < n) {
        vx[i] += DT * Fx;
        vy[i] += DT * Fy;
        vz[i] += DT * Fz;
        npx[i] = __fma_rn(vx[i], DT, xi);
        npy[i] = __fma_rn(vy[i], DT, yi);
        npz[i] = __fma_rn(vz[i], DT, zi);
    }
}

// Tile-parallel variant used when the body count is too small to fill the
// device with one thread per body. The expensive reciprocal-square-root work
// for a BI x BJ tile of pairs is spread over all threads; the summation for
// each body is then still performed in the original, strictly increasing j
// order, so the results are identical to the serial implementation.
template <int BI, int BJ, int BT>
__global__ void advanceBodiesTiledKernel(const double* __restrict__ px, const double* __restrict__ py,
                                         const double* __restrict__ pz, double* __restrict__ vx,
                                         double* __restrict__ vy, double* __restrict__ vz,
                                         double* __restrict__ npx, double* __restrict__ npy,
                                         double* __restrict__ npz, int n) {
    constexpr int BJS = BJ + 1;
    __shared__ double six[BI], siy[BI], siz[BI];
    __shared__ double sx[BJ], sy[BJ], sz[BJ];
    __shared__ double inv3[BI * BJS];

    const int lane = threadIdx.x;
    const int i0 = blockIdx.x * BI;

    for (int k = lane; k < BI; k += BT) {
        const int i = i0 + k;
        if (i < n) {
            six[k] = __ldg(px + i);
            siy[k] = __ldg(py + i);
            siz[k] = __ldg(pz + i);
        } else {
            six[k] = siy[k] = siz[k] = 0.0;
        }
    }
    __syncthreads();

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int base = 0; base < n; base += BJ) {
        for (int k = lane; k < BJ; k += BT) {
            const int t = base + k;
            if (t < n) {
                sx[k] = __ldg(px + t);
                sy[k] = __ldg(py + t);
                sz[k] = __ldg(pz + t);
            }
        }
        __syncthreads();

        const int jmax = min(BJ, n - base);

        for (int idx = lane; idx < BI * BJ; idx += BT) {
            const int ii = idx / BJ;
            const int jj = idx & (BJ - 1);
            if (jj < jmax) {
                const double dx = sx[jj] - six[ii];
                const double dy = sy[jj] - siy[ii];
                const double dz = sz[jj] - siz[ii];
                const double distSqr = __fma_rn(dz, dz, __fma_rn(dx, dx, dy * dy)) + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                inv3[ii * BJS + jj] = invDist * invDist * invDist;
            }
        }
        __syncthreads();

        if (lane < BI) {
            const double xi = six[lane];
            const double yi = siy[lane];
            const double zi = siz[lane];
            const double* row = inv3 + lane * BJS;
#pragma unroll 4
            for (int j = 0; j < jmax; ++j) {
                const double dx = sx[j] - xi;
                const double dy = sy[j] - yi;
                const double dz = sz[j] - zi;
                Fx = __fma_rn(dx, row[j], Fx);
                Fy = __fma_rn(dy, row[j], Fy);
                Fz = __fma_rn(dz, row[j], Fz);
            }
        }
        __syncthreads();
    }

    if (lane < BI) {
        const int i = i0 + lane;
        if (i < n) {
            vx[i] += DT * Fx;
            vy[i] += DT * Fy;
            vz[i] += DT * Fz;
            npx[i] = __fma_rn(vx[i], DT, six[lane]);
            npy[i] = __fma_rn(vy[i], DT, siy[lane]);
            npz[i] = __fma_rn(vz[i], DT, siz[lane]);
        }
    }
}

// Dispatch parameters for the tiled kernel: below the crossover the device is
// under-occupied by the one-thread-per-body mapping, so the tile-parallel
// variant is used instead.
constexpr int TILED_SMALL_BODIES = 1024;
constexpr int TILED_MAX_BODIES = 4096;

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

    // Initialize bodies on the host with the original RNG so that the
    // initial state is identical to the serial implementation.
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    const size_t n = bodies.size();
    const size_t bytes = n * sizeof(double);
    const size_t allocBytes = n ? bytes : sizeof(double);

    double* d_px[2] = {nullptr, nullptr};
    double* d_py[2] = {nullptr, nullptr};
    double* d_pz[2] = {nullptr, nullptr};
    double *d_vx = nullptr, *d_vy = nullptr, *d_vz = nullptr;
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMalloc(&d_px[b], allocBytes));
        CUDA_CHECK(cudaMalloc(&d_py[b], allocBytes));
        CUDA_CHECK(cudaMalloc(&d_pz[b], allocBytes));
    }
    CUDA_CHECK(cudaMalloc(&d_vx, allocBytes));
    CUDA_CHECK(cudaMalloc(&d_vy, allocBytes));
    CUDA_CHECK(cudaMalloc(&d_vz, allocBytes));

    std::vector<double> h_px(n), h_py(n), h_pz(n);
    std::vector<double> h_vx(n), h_vy(n), h_vz(n);
    for (size_t i = 0; i < n; ++i) {
        h_px[i] = bodies[i].pos.x;
        h_py[i] = bodies[i].pos.y;
        h_pz[i] = bodies[i].pos.z;
        h_vx[i] = bodies[i].vel.x;
        h_vy[i] = bodies[i].vel.y;
        h_vz[i] = bodies[i].vel.z;
    }

    if (n > 0) {
        // Warm up the exact kernel that will be timed so that module loading
        // and first-launch overhead are not part of the measured simulation.
        if (numBodies <= TILED_SMALL_BODIES) {
            const int warmBlocks = (numBodies + 15) / 16;
            advanceBodiesTiledKernel<16, 128, 256><<<warmBlocks, 256>>>(
                d_px[0], d_py[0], d_pz[0], d_vx, d_vy, d_vz,
                d_px[1], d_py[1], d_pz[1], numBodies);
        } else if (numBodies <= TILED_MAX_BODIES) {
            const int warmBlocks = (numBodies + 31) / 32;
            advanceBodiesTiledKernel<32, 128, 256><<<warmBlocks, 256>>>(
                d_px[0], d_py[0], d_pz[0], d_vx, d_vy, d_vz,
                d_px[1], d_py[1], d_pz[1], numBodies);
        } else {
            const int warmBlocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
            advanceBodiesKernel<<<warmBlocks, BLOCK_SIZE>>>(
                d_px[0], d_py[0], d_pz[0], d_vx, d_vy, d_vz,
                d_px[1], d_py[1], d_pz[1], numBodies);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());

        CUDA_CHECK(cudaMemcpy(d_px[0], h_px.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_py[0], h_py.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pz[0], h_pz.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vx, h_vx.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vy, h_vy.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_vz, h_vz.data(), bytes, cudaMemcpyHostToDevice));
    }

    // Run simulation
    auto start = std::chrono::high_resolution_clock::now();

    int cur = 0;
    if (numBodies > 0) {
        // Pick the mapping that keeps the GPU occupied: tiled pair-parallel
        // kernels for small problem sizes, one thread per body otherwise.
        if (numBodies <= TILED_SMALL_BODIES) {
            const int tiledBlocks = (numBodies + 15) / 16;
            for (int step = 0; step < numSteps; ++step) {
                const int nxt = cur ^ 1;
                advanceBodiesTiledKernel<16, 128, 256><<<tiledBlocks, 256>>>(
                    d_px[cur], d_py[cur], d_pz[cur], d_vx, d_vy, d_vz,
                    d_px[nxt], d_py[nxt], d_pz[nxt], numBodies);
                cur = nxt;
            }
        } else if (numBodies <= TILED_MAX_BODIES) {
            const int tiledBlocks = (numBodies + 31) / 32;
            for (int step = 0; step < numSteps; ++step) {
                const int nxt = cur ^ 1;
                advanceBodiesTiledKernel<32, 128, 256><<<tiledBlocks, 256>>>(
                    d_px[cur], d_py[cur], d_pz[cur], d_vx, d_vy, d_vz,
                    d_px[nxt], d_py[nxt], d_pz[nxt], numBodies);
                cur = nxt;
            }
        } else {
            const int blocks = (numBodies + BLOCK_SIZE - 1) / BLOCK_SIZE;
            for (int step = 0; step < numSteps; ++step) {
                const int nxt = cur ^ 1;
                advanceBodiesKernel<<<blocks, BLOCK_SIZE>>>(
                    d_px[cur], d_py[cur], d_pz[cur], d_vx, d_vy, d_vz,
                    d_px[nxt], d_py[nxt], d_pz[nxt], numBodies);
                cur = nxt;
            }
        }
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    if (n > 0) {
        CUDA_CHECK(cudaMemcpy(h_px.data(), d_px[cur], bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_py.data(), d_py[cur], bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_pz.data(), d_pz[cur], bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vx.data(), d_vx, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vy.data(), d_vy, bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(h_vz.data(), d_vz, bytes, cudaMemcpyDeviceToHost));
    }

    for (size_t i = 0; i < n; ++i) {
        bodies[i].pos.x = h_px[i];
        bodies[i].pos.y = h_py[i];
        bodies[i].pos.z = h_pz[i];
        bodies[i].vel.x = h_vx[i];
        bodies[i].vel.y = h_vy[i];
        bodies[i].vel.z = h_vz[i];
    }

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFree(d_px[b]));
        CUDA_CHECK(cudaFree(d_py[b]));
        CUDA_CHECK(cudaFree(d_pz[b]));
    }
    CUDA_CHECK(cudaFree(d_vx));
    CUDA_CHECK(cudaFree(d_vy));
    CUDA_CHECK(cudaFree(d_vz));

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
