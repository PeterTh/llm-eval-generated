#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

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

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, 1);                                                  \
        }                                                                                  \
    } while (0)

constexpr int BX = 64;  // bodies per block
constexpr int BY = 4;   // j-range splits per body
constexpr int TILE = BX * BY;

// pos: all bodies, interleaved xyz. vel: local slice (xyz). Slice starts at global index `off`, has `cnt` bodies.
__global__ void forcesKernel(const double* __restrict__ pos, double* __restrict__ vel, int n, int off, int cnt) {
    __shared__ double sp[TILE * 3];
    __shared__ double red[BY][BX][3];
    const int tx = threadIdx.x, ty = threadIdx.y;
    const int tid = ty * BX + tx;
    const int li = blockIdx.x * BX + tx;
    const int gi = off + li;
    double xi = 0, yi = 0, zi = 0;
    if (li < cnt) {
        xi = pos[3 * gi]; yi = pos[3 * gi + 1]; zi = pos[3 * gi + 2];
    }
    double Fx = 0.0, Fy = 0.0, Fz = 0.0;
    for (int base = 0; base < n; base += TILE) {
        const int j = base + tid;
        if (j < n) {
            sp[3 * tid] = pos[3 * j]; sp[3 * tid + 1] = pos[3 * j + 1]; sp[3 * tid + 2] = pos[3 * j + 2];
        } else {
            // far-away padding body is avoided by bounding the loop below
            sp[3 * tid] = sp[3 * tid + 1] = sp[3 * tid + 2] = 0.0;
        }
        __syncthreads();
        int lo = ty * BX;
        int hi = min(lo + BX, n - base);
        #pragma unroll 4
        for (int k = lo; k < hi; ++k) {
            const double dx = sp[3 * k] - xi;
            const double dy = sp[3 * k + 1] - yi;
            const double dz = sp[3 * k + 2] - zi;
            const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
            const double invDist = 1.0 / sqrt(distSqr);
            const double invDist3 = invDist * invDist * invDist;
            Fx += dx * invDist3;
            Fy += dy * invDist3;
            Fz += dz * invDist3;
        }
        __syncthreads();
    }
    red[ty][tx][0] = Fx; red[ty][tx][1] = Fy; red[ty][tx][2] = Fz;
    __syncthreads();
    if (ty == 0 && li < cnt) {
        for (int y = 1; y < BY; ++y) {
            Fx += red[y][tx][0]; Fy += red[y][tx][1]; Fz += red[y][tx][2];
        }
        vel[3 * li] += DT * Fx;
        vel[3 * li + 1] += DT * Fy;
        vel[3 * li + 2] += DT * Fz;
    }
}

// Integrate local slice and write new positions into the full position array (device).
__global__ void integrateKernel(double* __restrict__ pos, const double* __restrict__ vel, int off, int cnt) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li < cnt) {
        for (int c = 0; c < 3; ++c) pos[3 * (off + li) + c] += vel[3 * li + c] * DT;
    }
}

// Per-body potential energy contributions (pairs j > i) for local slice.
__global__ void energyKernel(const double* __restrict__ pos, double* __restrict__ out, int n, int off, int cnt) {
    int li = blockIdx.x * blockDim.x + threadIdx.x;
    if (li >= cnt) return;
    const int gi = off + li;
    const double xi = pos[3 * gi], yi = pos[3 * gi + 1], zi = pos[3 * gi + 2];
    double e = 0.0;
    for (int j = gi + 1; j < n; ++j) {
        const double dx = pos[3 * j] - xi, dy = pos[3 * j + 1] - yi, dz = pos[3 * j + 2] - zi;
        e -= 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
    }
    out[li] = e;
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

static double computePotentialEnergy(const double* d_pos, int n, int off, int cnt) {
    double local = 0.0;
    if (cnt > 0) {
        double* d_out;
        CUDA_CHECK(cudaMalloc(&d_out, cnt * sizeof(double)));
        energyKernel<<<(cnt + 127) / 128, 128>>>(d_pos, d_out, n, off, cnt);
        std::vector<double> h(cnt);
        CUDA_CHECK(cudaMemcpy(h.data(), d_out, cnt * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaFree(d_out));
        #pragma omp parallel for reduction(+ : local)
        for (int i = 0; i < cnt; ++i) local += h[i];
    }
    double total = 0.0;
    MPI_Allreduce(&local, &total, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

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
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (numBodies < 0) numBodies = 0;

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    // Select GPU by node-local rank
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int lrank;
        MPI_Comm_rank(local, &lrank);
        MPI_Comm_free(&local);
        int ndev = 0;
        CUDA_CHECK(cudaGetDeviceCount(&ndev));
        CUDA_CHECK(cudaSetDevice(lrank % ndev));
    }

    // Initialize bodies (identical on all ranks)
    std::vector<Body> bodies(numBodies);
    randomizeBodies(bodies);

    // Block distribution
    std::vector<int> counts(size), displs(size), counts3(size), displs3(size);
    for (int r = 0, o = 0; r < size; ++r) {
        counts[r] = numBodies / size + (r < numBodies % size ? 1 : 0);
        displs[r] = o;
        counts3[r] = counts[r] * 3;
        displs3[r] = o * 3;
        o += counts[r];
    }
    const int off = displs[rank], cnt = counts[rank];

    std::vector<double> hPos(3 * (size_t)numBodies + 1), hVel(3 * (size_t)cnt + 1);
    for (int i = 0; i < numBodies; ++i) {
        hPos[3 * i] = bodies[i].pos.x; hPos[3 * i + 1] = bodies[i].pos.y; hPos[3 * i + 2] = bodies[i].pos.z;
    }
    for (int i = 0; i < cnt; ++i) {
        hVel[3 * i] = bodies[off + i].vel.x; hVel[3 * i + 1] = bodies[off + i].vel.y; hVel[3 * i + 2] = bodies[off + i].vel.z;
    }

    double *d_pos, *d_vel;
    CUDA_CHECK(cudaMalloc(&d_pos, hPos.size() * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_vel, hVel.size() * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_pos, hPos.data(), hPos.size() * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_vel, hVel.data(), hVel.size() * sizeof(double), cudaMemcpyHostToDevice));

    const dim3 fblock(BX, BY);
    const int fgrid = (cnt + BX - 1) / BX;
    const int igrid = (cnt + 255) / 256;

    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    for (int step = 0; step < numSteps; ++step) {
        if (cnt > 0) {
            forcesKernel<<<fgrid, fblock>>>(d_pos, d_vel, numBodies, off, cnt);
            integrateKernel<<<igrid, 256>>>(d_pos, d_vel, off, cnt);
            CUDA_CHECK(cudaMemcpy(hPos.data() + 3 * (size_t)off, d_pos + 3 * (size_t)off,
                                  3 * (size_t)cnt * sizeof(double), cudaMemcpyDeviceToHost));
        }
        if (size > 1) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hPos.data(), counts3.data(), displs3.data(),
                           MPI_DOUBLE, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(d_pos, hPos.data(), 3 * (size_t)numBodies * sizeof(double),
                                  cudaMemcpyHostToDevice));
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Barrier(MPI_COMM_WORLD);
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    long local_ms = duration.count();
    long max_ms = 0;
    MPI_Reduce(&local_ms, &max_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) printf("Simulation time: %ld ms\n", max_ms);

    int rc = 0;
    if (printResults || validate) {
        // Gather final state on all ranks
        if (cnt > 0)
            CUDA_CHECK(cudaMemcpy(hVel.data(), d_vel, 3 * (size_t)cnt * sizeof(double), cudaMemcpyDeviceToHost));
        std::vector<double> allVel(3 * (size_t)numBodies + 1);
        MPI_Allgatherv(hVel.data(), 3 * cnt, MPI_DOUBLE, allVel.data(), counts3.data(), displs3.data(),
                       MPI_DOUBLE, MPI_COMM_WORLD);
        #pragma omp parallel for
        for (int i = 0; i < numBodies; ++i) {
            bodies[i].pos = Vec3(hPos[3 * i], hPos[3 * i + 1], hPos[3 * i + 2]);
            bodies[i].vel = Vec3(allVel[3 * i], allVel[3 * i + 1], allVel[3 * i + 2]);
        }

        if (printResults && rank == 0) {
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

        if (validate) {
            if (rank == 0) printf("Validating simulation results...\n");
            bool ok = validateSimulation(bodies);  // identical on every rank
            if (ok) {
                double energy = 0.0;
                #pragma omp parallel for reduction(+ : energy)
                for (int i = 0; i < numBodies; ++i) {
                    const Body& b = bodies[i];
                    energy += 0.5 * (b.vel.x * b.vel.x + b.vel.y * b.vel.y + b.vel.z * b.vel.z);
                }
                energy += computePotentialEnergy(d_pos, numBodies, off, cnt);
                if (rank == 0) {
                    printf("Final energy: %.6f\n", energy);
                    printf("Validation: PASSED\n");
                }
            } else {
                if (rank == 0) printf("Validation: FAILED\n");
                rc = 1;
            }
        }
    }

    CUDA_CHECK(cudaFree(d_pos));
    CUDA_CHECK(cudaFree(d_vel));
    MPI_Finalize();
    return rc;
}
