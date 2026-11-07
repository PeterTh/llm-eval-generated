// N-body benchmark, CUDA implementation.
//
// The all-pairs force computation (O(n^2)) runs on the GPU as a tiled
// shared-memory kernel in double precision. Body state is kept on the device in
// structure-of-arrays layout for the whole simulation; the host only touches it
// for initialization and final output. The j-loop is additionally split across
// several blocks (grid.y) so that small body counts still fill the whole GPU;
// the partial sums are combined in a fixed order, so results are deterministic.

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

#define CUDA_CHECK(call)                                                                       \
    do {                                                                                       \
        const cudaError_t err__ = (call);                                                      \
        if (err__ != cudaSuccess) {                                                            \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__, __LINE__,         \
                    cudaGetErrorString(err__));                                                \
            exit(EXIT_FAILURE);                                                                \
        }                                                                                      \
    } while (0)

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

// ---------------------------------------------------------------------------
// Device data layout (structure of arrays)
// ---------------------------------------------------------------------------
struct DeviceBodies {
    double* px = nullptr;
    double* py = nullptr;
    double* pz = nullptr;
    double* vx = nullptr;
    double* vy = nullptr;
    double* vz = nullptr;
    int n = 0;
};

constexpr int TILE = 256;  // threads per block == bodies per shared-memory tile

// ---------------------------------------------------------------------------
// Force kernel: thread i accumulates the force on body i over the j-range
// [jBegin, jEnd) assigned to blockIdx.y, tile by tile through shared memory.
// Partial forces are stored per (split, body) and combined by integrateKernel.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE)
forceKernel(const double* __restrict__ px, const double* __restrict__ py, const double* __restrict__ pz,
            double* __restrict__ fx, double* __restrict__ fy, double* __restrict__ fz,
            const int n, const int jChunk) {
    __shared__ double sx[TILE];
    __shared__ double sy[TILE];
    __shared__ double sz[TILE];

    const int i = blockIdx.x * TILE + threadIdx.x;
    const int jBegin = blockIdx.y * jChunk;
    const int jEnd = min(jBegin + jChunk, n);

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) {
        xi = px[i];
        yi = py[i];
        zi = pz[i];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (int tileStart = jBegin; tileStart < jEnd; tileStart += TILE) {
        const int j = tileStart + threadIdx.x;
        if (j < jEnd) {
            sx[threadIdx.x] = px[j];
            sy[threadIdx.x] = py[j];
            sz[threadIdx.x] = pz[j];
        }
        __syncthreads();

        const int tileCount = min(TILE, jEnd - tileStart);
        if (tileCount == TILE) {
            // Full tile: fixed trip count lets the compiler unroll aggressively.
#pragma unroll 16
            for (int k = 0; k < TILE; ++k) {
                const double dx = sx[k] - xi;
                const double dy = sy[k] - yi;
                const double dz = sz[k] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                // Double-precision rsqrt is accurate to 1 ulp, matching 1.0/sqrt().
                const double invDist = rsqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        } else {
#pragma unroll 4
            for (int k = 0; k < tileCount; ++k) {
                const double dx = sx[k] - xi;
                const double dy = sy[k] - yi;
                const double dz = sz[k] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                // Double-precision rsqrt is accurate to 1 ulp, matching 1.0/sqrt().
                const double invDist = rsqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;
                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (i < n) {
        const size_t slot = (size_t)blockIdx.y * n + i;
        fx[slot] = Fx;
        fy[slot] = Fy;
        fz[slot] = Fz;
    }
}

// ---------------------------------------------------------------------------
// Integration kernel: combine the j-split partial forces in a fixed order,
// update velocity, then advance position (same ordering as the original
// computeForces() followed by integrateBodies()).
// ---------------------------------------------------------------------------
__global__ void integrateKernel(double* __restrict__ px, double* __restrict__ py, double* __restrict__ pz,
                                double* __restrict__ vx, double* __restrict__ vy, double* __restrict__ vz,
                                const double* __restrict__ fx, const double* __restrict__ fy,
                                const double* __restrict__ fz, const int n, const int splits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int s = 0; s < splits; ++s) {
        const size_t slot = (size_t)s * n + i;
        Fx += fx[slot];
        Fy += fy[slot];
        Fz += fz[slot];
    }

    const double nvx = vx[i] + DT * Fx;
    const double nvy = vy[i] + DT * Fy;
    const double nvz = vz[i] + DT * Fz;
    vx[i] = nvx;
    vy[i] = nvy;
    vz[i] = nvz;
    px[i] += nvx * DT;
    py[i] += nvy * DT;
    pz[i] += nvz * DT;
}

// ---------------------------------------------------------------------------
// Potential-energy kernel: thread i sums -1/dist over j > i; block-reduced
// partials are written out and summed on the host in a fixed order.
// ---------------------------------------------------------------------------
__global__ void __launch_bounds__(TILE)
potentialEnergyKernel(const double* __restrict__ px, const double* __restrict__ py,
                      const double* __restrict__ pz, double* __restrict__ blockSums, const int n) {
    __shared__ double sx[TILE];
    __shared__ double sy[TILE];
    __shared__ double sz[TILE];
    __shared__ double red[TILE];

    const int i = blockIdx.x * TILE + threadIdx.x;
    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (i < n) {
        xi = px[i];
        yi = py[i];
        zi = pz[i];
    }

    double e = 0.0;
    // Only tiles at or after this block's own tile can contain j > i.
    for (int tileStart = blockIdx.x * TILE; tileStart < n; tileStart += TILE) {
        const int j = tileStart + threadIdx.x;
        if (j < n) {
            sx[threadIdx.x] = px[j];
            sy[threadIdx.x] = py[j];
            sz[threadIdx.x] = pz[j];
        }
        __syncthreads();

        const int tileCount = min(TILE, n - tileStart);
        if (i < n) {
            for (int k = 0; k < tileCount; ++k) {
                if (tileStart + k > i) {
                    const double dx = sx[k] - xi;
                    const double dy = sy[k] - yi;
                    const double dz = sz[k] - zi;
                    e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
                }
            }
        }
        __syncthreads();
    }

    red[threadIdx.x] = e;
    __syncthreads();
    for (int stride = TILE / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) red[threadIdx.x] += red[threadIdx.x + stride];
        __syncthreads();
    }
    if (threadIdx.x == 0) blockSums[blockIdx.x] = red[0];
}

// ---------------------------------------------------------------------------
// Host-side wrappers
// ---------------------------------------------------------------------------
static inline int divUp(int a, int b) { return (a + b - 1) / b; }

DeviceBodies allocDeviceBodies(int n) {
    DeviceBodies d;
    d.n = n;
    const size_t bytes = (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d.px, bytes));
    CUDA_CHECK(cudaMalloc(&d.py, bytes));
    CUDA_CHECK(cudaMalloc(&d.pz, bytes));
    CUDA_CHECK(cudaMalloc(&d.vx, bytes));
    CUDA_CHECK(cudaMalloc(&d.vy, bytes));
    CUDA_CHECK(cudaMalloc(&d.vz, bytes));
    return d;
}

void freeDeviceBodies(DeviceBodies& d) {
    cudaFree(d.px);
    cudaFree(d.py);
    cudaFree(d.pz);
    cudaFree(d.vx);
    cudaFree(d.vy);
    cudaFree(d.vz);
    d = DeviceBodies{};
}

void uploadBodies(const std::vector<Body>& bodies, const DeviceBodies& d) {
    const int n = d.n;
    std::vector<double> h((size_t)n * 6);
    for (int i = 0; i < n; ++i) {
        h[0 * n + i] = bodies[i].pos.x;
        h[1 * n + i] = bodies[i].pos.y;
        h[2 * n + i] = bodies[i].pos.z;
        h[3 * n + i] = bodies[i].vel.x;
        h[4 * n + i] = bodies[i].vel.y;
        h[5 * n + i] = bodies[i].vel.z;
    }
    const size_t bytes = (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMemcpy(d.px, h.data() + 0 * n, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.py, h.data() + 1 * n, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.pz, h.data() + 2 * n, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vx, h.data() + 3 * n, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vy, h.data() + 4 * n, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d.vz, h.data() + 5 * n, bytes, cudaMemcpyHostToDevice));
}

void downloadBodies(const DeviceBodies& d, std::vector<Body>& bodies) {
    const int n = d.n;
    std::vector<double> h((size_t)n * 6);
    const size_t bytes = (size_t)n * sizeof(double);
    CUDA_CHECK(cudaMemcpy(h.data() + 0 * n, d.px, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.data() + 1 * n, d.py, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.data() + 2 * n, d.pz, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.data() + 3 * n, d.vx, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.data() + 4 * n, d.vy, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.data() + 5 * n, d.vz, bytes, cudaMemcpyDeviceToHost));
    for (int i = 0; i < n; ++i) {
        bodies[i].pos.x = h[0 * n + i];
        bodies[i].pos.y = h[1 * n + i];
        bodies[i].pos.z = h[2 * n + i];
        bodies[i].vel.x = h[3 * n + i];
        bodies[i].vel.y = h[4 * n + i];
        bodies[i].vel.z = h[5 * n + i];
    }
}

// Choose how many ways to split the j-loop so that the grid has enough blocks
// to occupy every SM several times over, even for small body counts.
int chooseSplits(int n, int numSMs) {
    if (n <= 0) return 1;
    const int blocksI = divUp(n, TILE);
    const int targetBlocks = numSMs * 8;
    int splits = divUp(targetBlocks, blocksI);
    const int maxSplits = std::max(1, divUp(n, 32));  // keep >= 32 bodies per j-chunk
    if (splits > maxSplits) splits = maxSplits;
    if (splits < 1) splits = 1;
    return splits;
}

struct Simulation {
    DeviceBodies d;
    double* fx = nullptr;
    double* fy = nullptr;
    double* fz = nullptr;
    int splits = 1;
    int jChunk = 0;
    dim3 forceGrid;
    int integrateBlocks = 0;
};

Simulation createSimulation(int n, int numSMs) {
    Simulation s;
    s.d = allocDeviceBodies(n);
    s.splits = chooseSplits(n, numSMs);
    // Make the chunk a multiple of TILE where possible so full tiles dominate.
    s.jChunk = std::max(1, divUp(n, s.splits));
    if (s.jChunk > TILE) s.jChunk = divUp(s.jChunk, TILE) * TILE;
    s.splits = std::max(1, divUp(n, s.jChunk));
    const size_t fbytes = (size_t)s.splits * n * sizeof(double);
    CUDA_CHECK(cudaMalloc(&s.fx, fbytes));
    CUDA_CHECK(cudaMalloc(&s.fy, fbytes));
    CUDA_CHECK(cudaMalloc(&s.fz, fbytes));
    s.forceGrid = dim3(divUp(n, TILE), s.splits);
    s.integrateBlocks = divUp(n, TILE);
    return s;
}

void destroySimulation(Simulation& s) {
    cudaFree(s.fx);
    cudaFree(s.fy);
    cudaFree(s.fz);
    freeDeviceBodies(s.d);
    s = Simulation{};
}

void stepSimulation(const Simulation& s) {
    const DeviceBodies& d = s.d;
    if (d.n == 0) return;
    forceKernel<<<s.forceGrid, TILE>>>(d.px, d.py, d.pz, s.fx, s.fy, s.fz, d.n, s.jChunk);
    integrateKernel<<<s.integrateBlocks, TILE>>>(d.px, d.py, d.pz, d.vx, d.vy, d.vz, s.fx, s.fy, s.fz, d.n,
                                                 s.splits);
}

double computeTotalEnergy(const std::vector<Body>& bodies, const Simulation& s) {
    double energy = 0.0;
    const int n = (int)bodies.size();

    // Kinetic energy (assuming unit mass)
    for (const auto& body : bodies) {
        energy += 0.5 * (body.vel.x * body.vel.x + body.vel.y * body.vel.y + body.vel.z * body.vel.z);
    }

    // Potential energy (assuming unit mass for all bodies), pairs i<j on the GPU
    const int blocks = divUp(n, TILE);
    if (blocks == 0) return energy;
    double* dBlockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&dBlockSums, (size_t)blocks * sizeof(double)));
    potentialEnergyKernel<<<blocks, TILE>>>(s.d.px, s.d.py, s.d.pz, dBlockSums, n);
    CUDA_CHECK(cudaGetLastError());
    std::vector<double> blockSums(blocks);
    CUDA_CHECK(cudaMemcpy(blockSums.data(), dBlockSums, (size_t)blocks * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dBlockSums));
    for (int b = 0; b < blocks; ++b) energy += blockSums[b];

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

    if (numBodies < 0) numBodies = 0;

    printf("N-Body Simulation\n");
    printf("Number of bodies: %d\n", numBodies);
    printf("Number of steps: %d\n", numSteps);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");

    // Select GPU and query its SM count for grid sizing
    int device = 0;
    CUDA_CHECK(cudaSetDevice(device));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    printf("CUDA device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    // Initialize bodies
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    Simulation sim = createSimulation(numBodies, prop.multiProcessorCount);

    // Force the kernels' modules to load now so that lazy loading does not
    // land inside the timed region.
    cudaFuncAttributes attr;
    CUDA_CHECK(cudaFuncGetAttributes(&attr, forceKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, integrateKernel));
    CUDA_CHECK(cudaFuncGetAttributes(&attr, potentialEnergyKernel));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Run simulation (timed region includes host<->device transfers)
    auto start = std::chrono::high_resolution_clock::now();

    uploadBodies(bodies, sim.d);
    for (int step = 0; step < numSteps; ++step) {
        stepSimulation(sim);
    }
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    downloadBodies(sim.d, bodies);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    printf("Simulation time: %ld ms\n", duration.count());

    // Print results for external validation
    if (printResults) {
        // Serialize body positions and velocities for hashing
        std::vector<double> bodyData;
        bodyData.reserve((size_t)numBodies * 6);
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

    int exitCode = 0;

    // Validation: check that simulation produces finite, reasonable values
    if (validate) {
        printf("Validating simulation results...\n");

        if (validateSimulation(bodies)) {
            // Report final energy for reference
            double finalEnergy = computeTotalEnergy(bodies, sim);
            printf("Final energy: %.6f\n", finalEnergy);
            printf("Validation: PASSED\n");
        } else {
            printf("Validation: FAILED\n");
            exitCode = 1;
        }
    }

    destroySimulation(sim);
    return exitCode;
}
