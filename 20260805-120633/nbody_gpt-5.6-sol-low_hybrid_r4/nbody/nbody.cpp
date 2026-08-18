#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 {
    double x, y, z;
    constexpr Vec3(double x = 0, double y = 0, double z = 0) noexcept : x(x), y(y), z(z) {}
};
struct Body { Vec3 pos; Vec3 vel; };

static void mpiCheck(int error, const char* operation) {
    if (error == MPI_SUCCESS) return;
    char message[MPI_MAX_ERROR_STRING]; int length = 0;
    MPI_Error_string(error, message, &length);
    std::fprintf(stderr, "%s failed: %.*s\n", operation, length, message);
    MPI_Abort(MPI_COMM_WORLD, error);
}

static void cudaCheck(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return;
    std::fprintf(stderr, "%s failed: %s\n", operation, cudaGetErrorString(error));
    MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    // Kept serial so the initial state is bit-for-bit identical to the original.
    for (auto& body : bodies) {
        body.pos.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.pos.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.x = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.y = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
        body.vel.z = 2.0 * (rand_r(&seed) / (double)RAND_MAX) - 1.0;
    }
}

__global__ void forceKernel(Body* bodies, int n, int first, int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    const int i = first + local;
    const Vec3 pi = bodies[i].pos;
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = bodies[j].pos.x - pi.x;
        const double dy = bodies[j].pos.y - pi.y;
        const double dz = bodies[j].pos.z - pi.z;
        const double inv = 1.0 / sqrt(dx * dx + dy * dy + dz * dz + SOFTENING);
        const double inv3 = inv * inv * inv;
        fx += dx * inv3; fy += dy * inv3; fz += dz * inv3;
    }
    bodies[i].vel.x += DT * fx;
    bodies[i].vel.y += DT * fy;
    bodies[i].vel.z += DT * fz;
}

__global__ void integrateKernel(Body* bodies, int first, int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    Body& b = bodies[first + local];
    b.pos.x += b.vel.x * DT; b.pos.y += b.vel.y * DT; b.pos.z += b.vel.z * DT;
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    const long long n = static_cast<long long>(bodies.size());
    double kinetic = 0.0, potential = 0.0;
#pragma omp parallel for reduction(+:kinetic) schedule(static)
    for (long long i = 0; i < n; ++i) {
        const auto& v = bodies[i].vel;
        kinetic += 0.5 * (v.x*v.x + v.y*v.y + v.z*v.z);
    }
#pragma omp parallel for reduction(+:potential) schedule(dynamic,8)
    for (long long i = 0; i < n; ++i)
        for (long long j = i + 1; j < n; ++j) {
            const double dx=bodies[j].pos.x-bodies[i].pos.x;
            const double dy=bodies[j].pos.y-bodies[i].pos.y;
            const double dz=bodies[j].pos.z-bodies[i].pos.z;
            potential -= 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + SOFTENING);
        }
    return kinetic + potential;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    int invalid = 0;
#pragma omp parallel for reduction(|:invalid) schedule(static)
    for (long long i = 0; i < static_cast<long long>(bodies.size()); ++i) {
        const Body& b = bodies[i];
        invalid |= !std::isfinite(b.pos.x) || !std::isfinite(b.pos.y) || !std::isfinite(b.pos.z) ||
                   !std::isfinite(b.vel.x) || !std::isfinite(b.vel.y) || !std::isfinite(b.vel.z) ||
                   std::abs(b.pos.x)>1e6 || std::abs(b.pos.y)>1e6 || std::abs(b.pos.z)>1e6 ||
                   std::abs(b.vel.x)>1e6 || std::abs(b.vel.y)>1e6 || std::abs(b.vel.z)>1e6;
    }
    if (invalid) std::printf("Validation failed: found a non-finite or extreme body state\n");
    return !invalid;
}

void printUsage(const char* p) {
    std::printf("Usage: %s [options]\n  -n <num>  Number of bodies (default: 1024)\n"
                "  -s <num>  Number of steps (default: 10)\n  -v  Enable validation\n"
                "  -r  Print results for external validation\n  -h  Show this help message\n", p);
}

int main(int argc, char** argv) {
    int provided = 0;
    mpiCheck(MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided), "MPI_Init_thread");
    int rank = 0, ranks = 1;
    mpiCheck(MPI_Comm_rank(MPI_COMM_WORLD, &rank), "MPI_Comm_rank");
    mpiCheck(MPI_Comm_size(MPI_COMM_WORLD, &ranks), "MPI_Comm_size");

    int numBodies=1024, numSteps=10; bool validate=false, printResults=false; int parseStatus=0;
    for (int i=1; i<argc; ++i) {
        if (!std::strcmp(argv[i],"-n") && i+1<argc) numBodies=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-s") && i+1<argc) numSteps=std::atoi(argv[++i]);
        else if (!std::strcmp(argv[i],"-v")) validate=true;
        else if (!std::strcmp(argv[i],"-r")) printResults=true;
        else if (!std::strcmp(argv[i],"-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) { std::printf("Unknown option: %s\n",argv[i]); printUsage(argv[0]); } parseStatus=1; }
    }
    if (parseStatus || numBodies < 1 || numSteps < 0) {
        if (rank==0 && !parseStatus) std::fprintf(stderr,"Body count must be positive and steps non-negative\n");
        MPI_Finalize(); return 1;
    }

    // Bind ranks round-robin to the GPUs visible to each node.
    MPI_Comm localComm;
    mpiCheck(MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &localComm), "MPI_Comm_split_type");
    int localRank=0, devices=0; MPI_Comm_rank(localComm,&localRank);
    cudaCheck(cudaGetDeviceCount(&devices), "cudaGetDeviceCount");
    if (!devices) { if(rank==0) std::fprintf(stderr,"CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD,1); }
    cudaCheck(cudaSetDevice(localRank % devices), "cudaSetDevice");
    MPI_Comm_free(&localComm);

    const int base=numBodies/ranks, extra=numBodies%ranks;
    const int localCount=base+(rank<extra), first=rank*base+(rank<extra?rank:extra);
    std::vector<int> byteCounts(ranks), byteOffsets(ranks);
    for (int r=0; r<ranks; ++r) {
        const int c=base+(r<extra), f=r*base+(r<extra?r:extra);
        byteCounts[r]=c*static_cast<int>(sizeof(Body)); byteOffsets[r]=f*static_cast<int>(sizeof(Body));
    }
    std::vector<Body> bodies(numBodies);
    if(rank==0) randomizeBodies(bodies);
    mpiCheck(MPI_Bcast(bodies.data(), numBodies*sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD), "MPI_Bcast");

    Body* deviceBodies=nullptr; Body* exchange=nullptr;
    cudaCheck(cudaMalloc(&deviceBodies, numBodies*sizeof(Body)), "cudaMalloc");
    cudaCheck(cudaMallocHost(&exchange, numBodies*sizeof(Body)), "cudaMallocHost");
    cudaCheck(cudaMemcpy(deviceBodies,bodies.data(),numBodies*sizeof(Body),cudaMemcpyHostToDevice), "initial cudaMemcpy");
    if(rank==0) {
        std::printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",
                    numBodies,numSteps,validate?"enabled":"disabled");
        std::printf("Hybrid execution: %d MPI rank(s), %d OpenMP thread(s)/rank, CUDA\n",ranks,omp_get_max_threads());
    }
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "MPI_Barrier");
    const auto start=std::chrono::high_resolution_clock::now();
    constexpr int threads=256; const int blocks=(localCount+threads-1)/threads;
    for(int step=0; step<numSteps; ++step) {
        if(localCount) {
            forceKernel<<<blocks,threads>>>(deviceBodies,numBodies,first,localCount);
            integrateKernel<<<blocks,threads>>>(deviceBodies,first,localCount);
            cudaCheck(cudaGetLastError(), "CUDA kernel launch");
            cudaCheck(cudaMemcpy(exchange+first,deviceBodies+first,localCount*sizeof(Body),cudaMemcpyDeviceToHost), "device-to-host exchange");
        }
        mpiCheck(MPI_Allgatherv(exchange+first,byteCounts[rank],MPI_BYTE,exchange,byteCounts.data(),byteOffsets.data(),MPI_BYTE,MPI_COMM_WORLD), "MPI_Allgatherv");
        cudaCheck(cudaMemcpy(deviceBodies,exchange,numBodies*sizeof(Body),cudaMemcpyHostToDevice), "host-to-device exchange");
    }
    mpiCheck(MPI_Barrier(MPI_COMM_WORLD), "final MPI_Barrier");
    const auto end=std::chrono::high_resolution_clock::now();
    cudaCheck(cudaMemcpy(bodies.data(),deviceBodies,numBodies*sizeof(Body),cudaMemcpyDeviceToHost), "final cudaMemcpy");
    cudaFree(deviceBodies); cudaFreeHost(exchange);
    const long long localMs=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    long long elapsedMs=0; mpiCheck(MPI_Reduce(&localMs,&elapsedMs,1,MPI_LONG_LONG,MPI_MAX,0,MPI_COMM_WORLD), "MPI_Reduce");

    int status=0;
    if(rank==0) {
        std::printf("Simulation time: %lld ms\n",elapsedMs);
        if(printResults) {
            std::vector<double> data(static_cast<size_t>(numBodies)*6);
#pragma omp parallel for schedule(static)
            for(int i=0;i<numBodies;++i) {
                data[6*i]=bodies[i].pos.x; data[6*i+1]=bodies[i].pos.y; data[6*i+2]=bodies[i].pos.z;
                data[6*i+3]=bodies[i].vel.x; data[6*i+4]=bodies[i].vel.y; data[6*i+5]=bodies[i].vel.z;
            }
            print_results(data,"Bodies");
        }
        if(validate) {
            std::printf("Validating simulation results...\n");
            if(validateSimulation(bodies)) {
                std::printf("Final energy: %.6f\nValidation: PASSED\n",computeTotalEnergy(bodies));
            } else { std::printf("Validation: FAILED\n"); status=1; }
        }
    }
    mpiCheck(MPI_Bcast(&status,1,MPI_INT,0,MPI_COMM_WORLD), "status MPI_Bcast");
    MPI_Finalize(); return status;
}
