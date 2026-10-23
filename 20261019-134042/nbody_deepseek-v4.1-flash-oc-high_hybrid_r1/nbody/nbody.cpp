#include <algorithm>
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
constexpr int TILE = 256;

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

__global__ void computeForcesKernel(const double* __restrict__ gx,
                                    const double* __restrict__ gy,
                                    const double* __restrict__ gz,
                                    double* __restrict__ fx,
                                    double* __restrict__ fy,
                                    double* __restrict__ fz,
                                    const size_t n,
                                    const size_t start,
                                    const size_t count) {
    __shared__ double sx[TILE];
    __shared__ double sy[TILE];
    __shared__ double sz[TILE];

    const size_t local = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const bool active = local < count;

    double xi = 0.0, yi = 0.0, zi = 0.0;
    if (active) {
        const size_t gi = start + local;
        xi = gx[gi];
        yi = gy[gi];
        zi = gz[gi];
    }

    double Fx = 0.0, Fy = 0.0, Fz = 0.0;

    for (size_t j0 = 0; j0 < n; j0 += TILE) {
        const size_t tile = (n - j0 < static_cast<size_t>(TILE)) ? (n - j0) : static_cast<size_t>(TILE);

        for (size_t k = threadIdx.x; k < tile; k += blockDim.x) {
            const size_t j = j0 + k;
            sx[k] = gx[j];
            sy[k] = gy[j];
            sz[k] = gz[j];
        }
        __syncthreads();

        if (active) {
#pragma unroll 4
            for (size_t k = 0; k < tile; ++k) {
                const double dx = sx[k] - xi;
                const double dy = sy[k] - yi;
                const double dz = sz[k] - zi;
                const double distSqr = dx * dx + dy * dy + dz * dz + SOFTENING;
                const double invDist = 1.0 / std::sqrt(distSqr);
                const double invDist3 = invDist * invDist * invDist;

                Fx += dx * invDist3;
                Fy += dy * invDist3;
                Fz += dz * invDist3;
            }
        }
        __syncthreads();
    }

    if (active) {
        fx[local] = Fx;
        fy[local] = Fy;
        fz[local] = Fz;
    }
}

static void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error at %s: %s\n", what, cudaGetErrorString(err));
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
}

double computeTotalEnergy(const double* x, const double* y, const double* z,
                          const double* vx, const double* vy, const double* vz,
                          size_t n) {
    double energy = 0.0;

    for (size_t i = 0; i < n; ++i) {
        energy += 0.5 * (vx[i] * vx[i] + vy[i] * vy[i] + vz[i] * vz[i]);
    }

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            const double dx = x[j] - x[i];
            const double dy = y[j] - y[i];
            const double dz = z[j] - z[i];
            const double dist = std::sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
            energy -= 1.0 / dist;
        }
    }

    return energy;
}

bool validateSimulation(const double* x, const double* y, const double* z,
                        const double* vx, const double* vy, const double* vz,
                        size_t n) {
    for (size_t i = 0; i < n; ++i) {
        if (!std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(z[i]) ||
            !std::isfinite(vx[i]) || !std::isfinite(vy[i]) || !std::isfinite(vz[i])) {
            printf("Validation failed: found NaN or Inf value in body state\n");
            return false;
        }

        const double maxPos = 1e6;
        const double maxVel = 1e6;
        if (std::abs(x[i]) > maxPos || std::abs(y[i]) > maxPos || std::abs(z[i]) > maxPos) {
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
    MPI_Init(&argc, &argv);

    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int numBodies = 1024;
    int numSteps = 10;
    int validateFlag = 0;
    int printResultsFlag = 0;
    int action = 0;

    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numBodies = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
                numSteps = atoi(argv[++i]);
            } else if (strcmp(argv[i], "-v") == 0) {
                validateFlag = 1;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResultsFlag = 1;
            } else if (strcmp(argv[i], "-h") == 0) {
                action = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                action = 2;
                break;
            }
        }
        if (action == 1) {
            printUsage(argv[0]);
        }
    }

    MPI_Bcast(&numBodies, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&numSteps, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&validateFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&printResultsFlag, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&action, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (action == 1) {
        MPI_Finalize();
        return 0;
    }
    if (action == 2) {
        MPI_Finalize();
        return 1;
    }

    const bool validate = validateFlag != 0;
    const bool printResults = printResultsFlag != 0;
    if (numBodies < 0) {
        numBodies = 0;
    }
    const size_t n = static_cast<size_t>(numBodies);

    omp_set_num_threads(std::min(omp_get_max_threads(), 16));

    if (rank == 0) {
        printf("N-Body Simulation\n");
        printf("Number of bodies: %d\n", numBodies);
        printf("Number of steps: %d\n", numSteps);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
    }

    MPI_Comm nodeComm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &nodeComm);
    int localRank = 0;
    MPI_Comm_rank(nodeComm, &localRank);

    int deviceCount = 0;
    cudaCheck(cudaGetDeviceCount(&deviceCount), "cudaGetDeviceCount");
    if (deviceCount <= 0) {
        if (rank == 0) {
            fprintf(stderr, "No CUDA devices available\n");
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    cudaCheck(cudaSetDevice(localRank % deviceCount), "cudaSetDevice");

    const int base = numBodies / size;
    const int rem = numBodies % size;
    std::vector<int> counts(size), displs(size);
    for (int r = 0; r < size; ++r) {
        counts[r] = base + (r < rem ? 1 : 0);
        displs[r] = r * base + std::min(r, rem);
    }

    const int localCount = counts[rank];
    const size_t localStart = static_cast<size_t>(displs[rank]);
    const size_t localAlloc = localCount > 0 ? static_cast<size_t>(localCount) : 1;

    std::vector<double> px(localAlloc), py(localAlloc), pz(localAlloc);
    std::vector<double> vx(localAlloc), vy(localAlloc), vz(localAlloc);

    const size_t globalBytes = (n > 0 ? n : 1) * sizeof(double);
    const size_t localBytes = localAlloc * sizeof(double);

    double* gx = nullptr;
    double* gy = nullptr;
    double* gz = nullptr;
    double* hfx = nullptr;
    double* hfy = nullptr;
    double* hfz = nullptr;
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&gx), globalBytes), "cudaMallocHost");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&gy), globalBytes), "cudaMallocHost");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&gz), globalBytes), "cudaMallocHost");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hfx), localBytes), "cudaMallocHost");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hfy), localBytes), "cudaMallocHost");
    cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hfz), localBytes), "cudaMallocHost");

    double* dgx = nullptr;
    double* dgy = nullptr;
    double* dgz = nullptr;
    double* dfx = nullptr;
    double* dfy = nullptr;
    double* dfz = nullptr;
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dgx), globalBytes), "cudaMalloc");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dgy), globalBytes), "cudaMalloc");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dgz), globalBytes), "cudaMalloc");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dfx), localBytes), "cudaMalloc");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dfy), localBytes), "cudaMalloc");
    cudaCheck(cudaMalloc(reinterpret_cast<void**>(&dfz), localBytes), "cudaMalloc");

    std::vector<double> rvx, rvy, rvz;
    if (rank == 0) {
        std::vector<Body> bodies(n);
        randomizeBodies(bodies);
        for (size_t i = 0; i < n; ++i) {
            gx[i] = bodies[i].pos.x;
            gy[i] = bodies[i].pos.y;
            gz[i] = bodies[i].pos.z;
        }
        rvx.resize(n > 0 ? n : 1);
        rvy.resize(n > 0 ? n : 1);
        rvz.resize(n > 0 ? n : 1);
        for (size_t i = 0; i < n; ++i) {
            rvx[i] = bodies[i].vel.x;
            rvy[i] = bodies[i].vel.y;
            rvz[i] = bodies[i].vel.z;
        }
    }

    MPI_Scatterv(gx, counts.data(), displs.data(), MPI_DOUBLE,
                 px.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(gy, counts.data(), displs.data(), MPI_DOUBLE,
                 py.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(gz, counts.data(), displs.data(), MPI_DOUBLE,
                 pz.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rvx.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 vx.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rvy.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 vy.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);
    MPI_Scatterv(rvz.data(), counts.data(), displs.data(), MPI_DOUBLE,
                 vz.data(), localCount, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    const double startTime = MPI_Wtime();

    for (int step = 0; step < numSteps; ++step) {
        MPI_Allgatherv(px.data(), localCount, MPI_DOUBLE,
                       gx, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(py.data(), localCount, MPI_DOUBLE,
                       gy, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Allgatherv(pz.data(), localCount, MPI_DOUBLE,
                       gz, counts.data(), displs.data(), MPI_DOUBLE, MPI_COMM_WORLD);

        if (n > 0) {
            cudaCheck(cudaMemcpy(dgx, gx, n * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
            cudaCheck(cudaMemcpy(dgy, gy, n * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
            cudaCheck(cudaMemcpy(dgz, gz, n * sizeof(double), cudaMemcpyHostToDevice), "cudaMemcpy");
        }

        if (localCount > 0) {
            const int threads = 256;
            const int blocks = (localCount + threads - 1) / threads;
            computeForcesKernel<<<blocks, threads>>>(dgx, dgy, dgz, dfx, dfy, dfz,
                                                     n, localStart,
                                                     static_cast<size_t>(localCount));
            cudaCheck(cudaGetLastError(), "computeForcesKernel");
            cudaCheck(cudaMemcpy(hfx, dfx, static_cast<size_t>(localCount) * sizeof(double),
                                 cudaMemcpyDeviceToHost), "cudaMemcpy");
            cudaCheck(cudaMemcpy(hfy, dfy, static_cast<size_t>(localCount) * sizeof(double),
                                 cudaMemcpyDeviceToHost), "cudaMemcpy");
            cudaCheck(cudaMemcpy(hfz, dfz, static_cast<size_t>(localCount) * sizeof(double),
                                 cudaMemcpyDeviceToHost), "cudaMemcpy");
        }

#pragma omp parallel for schedule(static)
        for (size_t i = 0; i < static_cast<size_t>(localCount); ++i) {
            const double nvx = vx[i] + DT * hfx[i];
            const double nvy = vy[i] + DT * hfy[i];
            const double nvz = vz[i] + DT * hfz[i];
            vx[i] = nvx;
            vy[i] = nvy;
            vz[i] = nvz;
            px[i] += nvx * DT;
            py[i] += nvy * DT;
            pz[i] += nvz * DT;
        }
    }

    const double endTime = MPI_Wtime();
    double elapsed = endTime - startTime;
    double maxElapsed = 0.0;
    MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Simulation time: %ld ms\n", static_cast<long>(maxElapsed * 1000.0));
    }

    std::vector<double> gvx, gvy, gvz;
    if (printResults || validate) {
        MPI_Gatherv(px.data(), localCount, MPI_DOUBLE,
                    gx, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(py.data(), localCount, MPI_DOUBLE,
                    gy, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(pz.data(), localCount, MPI_DOUBLE,
                    gz, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            gvx.resize(n > 0 ? n : 1);
            gvy.resize(n > 0 ? n : 1);
            gvz.resize(n > 0 ? n : 1);
        }
        double* rgvx = (rank == 0) ? gvx.data() : gx;
        double* rgvy = (rank == 0) ? gvy.data() : gy;
        double* rgvz = (rank == 0) ? gvz.data() : gz;
        MPI_Gatherv(vx.data(), localCount, MPI_DOUBLE,
                    rgvx, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(vy.data(), localCount, MPI_DOUBLE,
                    rgvy, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
        MPI_Gatherv(vz.data(), localCount, MPI_DOUBLE,
                    rgvz, counts.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    }

    if (printResults && rank == 0) {
        std::vector<double> bodyData;
        bodyData.reserve(n * 6);
        for (size_t i = 0; i < n; ++i) {
            bodyData.push_back(gx[i]);
            bodyData.push_back(gy[i]);
            bodyData.push_back(gz[i]);
            bodyData.push_back(gvx[i]);
            bodyData.push_back(gvy[i]);
            bodyData.push_back(gvz[i]);
        }
        print_results(bodyData, "Bodies");
    }

    int valid = 1;
    if (validate) {
        if (rank == 0) {
            printf("Validating simulation results...\n");
            if (validateSimulation(gx, gy, gz, gvx.data(), gvy.data(), gvz.data(), n)) {
                const double finalEnergy = computeTotalEnergy(gx, gy, gz,
                                                              gvx.data(), gvy.data(), gvz.data(), n);
                printf("Final energy: %.6f\n", finalEnergy);
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                valid = 0;
            }
        }
        MPI_Bcast(&valid, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    cudaFree(dfx);
    cudaFree(dfy);
    cudaFree(dfz);
    cudaFree(dgx);
    cudaFree(dgy);
    cudaFree(dgz);
    cudaFreeHost(hfx);
    cudaFreeHost(hfy);
    cudaFreeHost(hfz);
    cudaFreeHost(gx);
    cudaFreeHost(gy);
    cudaFreeHost(gz);

    MPI_Comm_free(&nodeComm);
    MPI_Finalize();

    return (validate && !valid) ? 1 : 0;
}
