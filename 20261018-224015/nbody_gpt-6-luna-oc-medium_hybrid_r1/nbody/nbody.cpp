#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

__global__ void advanceBodies(const double* state, double* output, int n, int first, int count) {
    const int local = blockIdx.x * blockDim.x + threadIdx.x;
    if (local >= count) return;
    const int i = first + local;
    const double ix = state[6*i], iy = state[6*i+1], iz = state[6*i+2];
    double fx = 0.0, fy = 0.0, fz = 0.0;
    for (int j = 0; j < n; ++j) {
        const double dx = state[6*j] - ix;
        const double dy = state[6*j+1] - iy;
        const double dz = state[6*j+2] - iz;
        const double r2 = dx*dx + dy*dy + dz*dz + SOFTENING;
        const double inv = 1.0 / sqrt(r2);
        const double inv3 = inv*inv*inv;
        fx += dx*inv3; fy += dy*inv3; fz += dz*inv3;
    }
    const double vx = state[6*i+3] + DT*fx;
    const double vy = state[6*i+4] + DT*fy;
    const double vz = state[6*i+5] + DT*fz;
    output[6*local] = ix + vx*DT; output[6*local+1] = iy + vy*DT; output[6*local+2] = iz + vz*DT;
    output[6*local+3] = vx; output[6*local+4] = vy; output[6*local+5] = vz;
}

void printUsage(const char* name) {
    printf("Usage: %s [options]\nOptions:\n  -n <num>     Number of bodies (default: 1024)\n  -s <num>     Number of simulation steps (default: 10)\n  -v           Enable validation\n  -r           Print results for external validation\n  -h           Show this help message\n", name);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int numBodies = 1024, numSteps = 10;
    bool validate = false, printResults = false;
    for (int i=1; i<argc; ++i) {
        if (!strcmp(argv[i], "-n") && i+1<argc) numBodies=atoi(argv[++i]);
        else if (!strcmp(argv[i], "-s") && i+1<argc) numSteps=atoi(argv[++i]);
        else if (!strcmp(argv[i], "-v")) validate=true;
        else if (!strcmp(argv[i], "-r")) printResults=true;
        else if (!strcmp(argv[i], "-h")) { if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0; }
        else { if(rank==0) { printf("Unknown option: %s\n", argv[i]); printUsage(argv[0]); } MPI_Finalize(); return 1; }
    }
    if (numBodies < 0) numBodies=0;
    std::vector<Body> bodies(static_cast<size_t>(numBodies));
    unsigned int seed=42;
    for (auto& b : bodies) {
        b.pos.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; b.pos.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; b.pos.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        b.vel.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; b.vel.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; b.vel.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
    }
    int devices=0;
    if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) { if(rank==0) fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2); }
    cudaSetDevice(rank % devices);
    const int n=numBodies, first=(n*rank)/ranks, end=(n*(rank+1))/ranks, localCount=end-first;
    std::vector<int> counts(ranks), displs(ranks);
    for(int r=0;r<ranks;++r) { const int lo=(n*r)/ranks, hi=(n*(r+1))/ranks; counts[r]=(hi-lo)*6; displs[r]=lo*6; }
    std::vector<double> state(static_cast<size_t>(n)*6), local(static_cast<size_t>(localCount)*6), gathered(static_cast<size_t>(n)*6);
    double *deviceState=nullptr, *deviceOutput=nullptr;
    cudaMalloc(&deviceState, state.size()*sizeof(double)); cudaMalloc(&deviceOutput, local.size()*sizeof(double));
    if(rank==0) { printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",n,numSteps,validate?"enabled":"disabled"); }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto start=std::chrono::high_resolution_clock::now();
    for(int step=0;step<numSteps;++step) {
        #pragma omp parallel for
        for(int i=0;i<n;++i) { state[6*i]=bodies[i].pos.x; state[6*i+1]=bodies[i].pos.y; state[6*i+2]=bodies[i].pos.z; state[6*i+3]=bodies[i].vel.x; state[6*i+4]=bodies[i].vel.y; state[6*i+5]=bodies[i].vel.z; }
        cudaMemcpy(deviceState,state.data(),state.size()*sizeof(double),cudaMemcpyHostToDevice);
        if(localCount) { advanceBodies<<<(localCount+127)/128,128>>>(deviceState,deviceOutput,n,first,localCount); cudaMemcpy(local.data(),deviceOutput,local.size()*sizeof(double),cudaMemcpyDeviceToHost); }
        MPI_Allgatherv(local.data(),localCount*6,MPI_DOUBLE,gathered.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        #pragma omp parallel for
        for(int i=0;i<n;++i) { bodies[i].pos={gathered[6*i],gathered[6*i+1],gathered[6*i+2]}; bodies[i].vel={gathered[6*i+3],gathered[6*i+4],gathered[6*i+5]}; }
    }
    MPI_Barrier(MPI_COMM_WORLD);
    const auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::high_resolution_clock::now()-start);
    long localMs = duration.count();
    long globalMs = 0;
    MPI_Reduce(&localMs, &globalMs, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank==0) printf("Simulation time: %ld ms\n",globalMs);
    if(rank==0 && printResults) {
        std::vector<double> data; data.reserve(static_cast<size_t>(n)*6);
        for(const auto& b:bodies) { data.insert(data.end(),{b.pos.x,b.pos.y,b.pos.z,b.vel.x,b.vel.y,b.vel.z}); }
        print_results(data,"Bodies");
    }
    int valid=1;
    if(validate) {
        for(const auto& b:bodies) if(!std::isfinite(b.pos.x)||!std::isfinite(b.pos.y)||!std::isfinite(b.pos.z)||!std::isfinite(b.vel.x)||!std::isfinite(b.vel.y)||!std::isfinite(b.vel.z)||std::abs(b.pos.x)>1e6||std::abs(b.pos.y)>1e6||std::abs(b.pos.z)>1e6||std::abs(b.vel.x)>1e6||std::abs(b.vel.y)>1e6||std::abs(b.vel.z)>1e6) valid=0;
        int allValid; MPI_Allreduce(&valid,&allValid,1,MPI_INT,MPI_MIN,MPI_COMM_WORLD);
        if(rank==0) printf("Validating simulation results...\nValidation: %s\n",allValid?"PASSED":"FAILED");
        valid=allValid;
    }
    cudaFree(deviceState); cudaFree(deviceOutput);
    MPI_Finalize();
    return valid?0:1;
}
