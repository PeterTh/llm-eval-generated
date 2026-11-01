#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;

struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed = 42) {
    #pragma omp parallel for
    for (size_t i = 0; i < bodies.size(); ++i) {
        unsigned int s = seed + (unsigned int)i;
        bodies[i].pos.x = 2.0 * (rand_r(&s) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.y = 2.0 * (rand_r(&s) / (double)RAND_MAX) - 1.0;
        bodies[i].pos.z = 2.0 * (rand_r(&s) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.x = 2.0 * (rand_r(&s) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.y = 2.0 * (rand_r(&s) / (double)RAND_MAX) - 1.0;
        bodies[i].vel.z = 2.0 * (rand_r(&s) / (double)RAND_MAX) - 1.0;
    }
}

// Wait: original uses single sequential seed stream. Need preserve exact semantics for correctness/hash.
// Revert to sequential to match original output exactly.
// (placeholder replaced below)
__global__ void forceKernel(const double* px, const double* py, const double* pz,
                           double* vx, double* vy, double* vz,
                           int n, int start, int end) {
    int i = start + blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= end) return;
    double Fx=0,Fy=0,Fz=0;
    double pix=px[i], piy=py[i], piz=pz[i];
    for (int j=0;j<n;++j){
        double dx=px[j]-pix, dy=py[j]-piy, dz=pz[j]-piz;
        double distSqr=dx*dx+dy*dy+dz*dz+SOFTENING;
        double invDist=1.0/sqrt(distSqr);
        double invDist3=invDist*invDist*invDist;
        Fx+=dx*invDist3; Fy+=dy*invDist3; Fz+=dz*invDist3;
    }
    vx[i]+=DT*Fx; vy[i]+=DT*Fy; vz[i]+=DT*Fz;
}

void computeForces(std::vector<Body>& bodies, int rank, int size) {
    const int n=bodies.size();
    // arrays
    std::vector<double> px(n),py(n),pz(n),vx(n),vy(n),vz(n);
    for(int i=0;i<n;++i){px[i]=bodies[i].pos.x;py[i]=bodies[i].pos.y;pz[i]=bodies[i].pos.z;vx[i]=bodies[i].vel.x;vy[i]=bodies[i].vel.y;vz[i]=bodies[i].vel.z;}
    // broadcast positions (all have full anyway)
    // partition i
    int chunk=n/size;
    int start=rank*chunk;
    int end=(rank==size-1)?n:(rank+1)*chunk;
    // device
    double *dpx,*dpy,*dpz,*dvx,*dvy,*dvz;
    cudaMalloc(&dpx,n*sizeof(double)); cudaMalloc(&dpy,n*sizeof(double)); cudaMalloc(&dpz,n*sizeof(double));
    cudaMalloc(&dvx,n*sizeof(double)); cudaMalloc(&dvy,n*sizeof(double)); cudaMalloc(&dvz,n*sizeof(double));
    cudaMemcpy(dpx,px.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dpy,py.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dpz,pz.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dvx,vx.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dvy,vy.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dvz,vz.data(),n*sizeof(double),cudaMemcpyHostToDevice);
    int threads=256;
    int blocks=(end-start+threads-1)/threads;
    forceKernel<<<blocks,threads>>>(dpx,dpy,dpz,dvx,dvy,dvz,n,start,end);
    cudaDeviceSynchronize();
    cudaMemcpy(vx.data(),dvx,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(vy.data(),dvy,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaMemcpy(vz.data(),dvz,n*sizeof(double),cudaMemcpyDeviceToHost);
    cudaFree(dpx);cudaFree(dpy);cudaFree(dpz);cudaFree(dvx);cudaFree(dvy);cudaFree(dvz);
    // gather velocities across ranks: each rank updated subset; combine
    // Use MPI_Allreduce? velocities disjoint; use Allgatherv of updated chunks
    // Simpler: allreduce sum (others unchanged zero diff? no, all have full original vx; only subset changed)
    // Need combine: rank has full array but only subset updated; others stale original. Use max? 
    // Better: each rank sends its updated subset; gather.
    std::vector<int> counts(size), displs(size);
    // counts in doubles? We'll gather vx,vy,vz separately with element counts
    // Build counts of bodies per rank
    std::vector<int> rcnt(size), rdisp(size);
    for(int r=0;r<size;++r){ int s=r*chunk; int e=(r==size-1)?n:(r+1)*chunk; rcnt[r]=e-s; rdisp[r]=s; }
    std::vector<double> gx(n),gy(n),gz(n);
    MPI_Allgatherv(vx.data()+start, end-start, MPI_DOUBLE, gx.data(), rcnt.data(), rdisp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(vy.data()+start, end-start, MPI_DOUBLE, gy.data(), rcnt.data(), rdisp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    MPI_Allgatherv(vz.data()+start, end-start, MPI_DOUBLE, gz.data(), rcnt.data(), rdisp.data(), MPI_DOUBLE, MPI_COMM_WORLD);
    for(int i=0;i<n;++i){bodies[i].vel.x=gx[i];bodies[i].vel.y=gy[i];bodies[i].vel.z=gz[i];}
}

void integrateBodies(std::vector<Body>& bodies) {
    #pragma omp parallel for
    for(size_t i=0;i<bodies.size();++i){
        bodies[i].pos.x += bodies[i].vel.x * DT;
        bodies[i].pos.y += bodies[i].vel.y * DT;
        bodies[i].pos.z += bodies[i].vel.z * DT;
    }
}

double computeTotalEnergy(const std::vector<Body>& bodies) {
    double energy=0.0;
    const size_t n=bodies.size();
    #pragma omp parallel for reduction(+:energy)
    for(size_t i=0;i<n;++i) energy += 0.5*(bodies[i].vel.x*bodies[i].vel.x+bodies[i].vel.y*bodies[i].vel.y+bodies[i].vel.z*bodies[i].vel.z);
    // potential
    double pot=0.0;
    #pragma omp parallel for reduction(+:pot)
    for(size_t i=0;i<n;++i){
        for(size_t j=i+1;j<n;++j){
            double dx=bodies[j].pos.x-bodies[i].pos.x;
            double dy=bodies[j].pos.y-bodies[i].pos.y;
            double dz=bodies[j].pos.z-bodies[i].pos.z;
            double dist=std::sqrt(dx*dx+dy*dy+dz*dz+SOFTENING);
            pot -= 1.0/dist;
        }
    }
    return energy+pot;
}

bool validateSimulation(const std::vector<Body>& bodies) {
    for(const auto& body:bodies){
        if (!std::isfinite(body.pos.x)||!std::isfinite(body.pos.y)||!std::isfinite(body.pos.z)||
            !std::isfinite(body.vel.x)||!std::isfinite(body.vel.y)||!std::isfinite(body.vel.z)){printf("Validation failed: found NaN or Inf value in body state\n");return false;}
        const double maxPos=1e6,maxVel=1e6;
        if (std::abs(body.pos.x)>maxPos||std::abs(body.pos.y)>maxPos||std::abs(body.pos.z)>maxPos){printf("Validation failed: body position exceeds reasonable bounds\n");return false;}
        if (std::abs(body.vel.x)>maxVel||std::abs(body.vel.y)>maxVel||std::abs(body.vel.z)>maxVel){printf("Validation failed: body velocity exceeds reasonable bounds\n");return false;}
    }
    return true;
}

void printUsage(const char* progName){ /* same */ printf("Usage: %s [options]\n", progName); printf("Options:\n"); printf("  -n <num>     Number of bodies (default: 1024)\n"); printf("  -s <num>     Number of simulation steps (default: 10)\n"); printf("  -v           Enable validation (checks energy conservation)\n"); printf("  -r           Print results for external validation\n"); printf("  -h           Show this help message\n"); }

int main(int argc, char** argv){
    MPI_Init(&argc,&argv);
    int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    int numBodies=1024,numSteps=10; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){
        if(strcmp(argv[i],"-n")==0&&i+1<argc) numBodies=atoi(argv[++i]);
        else if(strcmp(argv[i],"-s")==0&&i+1<argc) numSteps=atoi(argv[++i]);
        else if(strcmp(argv[i],"-v")==0) validate=true;
        else if(strcmp(argv[i],"-r")==0) printResults=true;
        else if(strcmp(argv[i],"-h")==0){ if(rank==0) printUsage(argv[0]); MPI_Finalize(); return 0;}
        else { if(rank==0){printf("Unknown option: %s\n",argv[i]);printUsage(argv[0]);} MPI_Finalize(); return 1;}
    }
    if(rank==0){printf("N-Body Simulation\n");printf("Number of bodies: %d\n",numBodies);printf("Number of steps: %d\n",numSteps);printf("Validation: %s\n",validate?"enabled":"disabled");}
    std::vector<Body> bodies(numBodies);
    // original sequential randomize to preserve output
    {
        unsigned int seed=42;
        for(auto& body:bodies){
            body.pos.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
            body.pos.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
            body.pos.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
            body.vel.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
            body.vel.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
            body.vel.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        }
    }
    // broadcast bodies
    MPI_Bcast(bodies.data(), numBodies*sizeof(Body), MPI_BYTE, 0, MPI_COMM_WORLD);
    auto start=std::chrono::high_resolution_clock::now();
    for(int step=0;step<numSteps;++step){
        computeForces(bodies,rank,size);
        // integrate on all ranks identically? integrate subset? We can integrate all on rank0 only? 
        // All ranks have full array; integrate everywhere consistently
        integrateBodies(bodies);
        // sync
        MPI_Barrier(MPI_COMM_WORLD);
    }
    auto end=std::chrono::high_resolution_clock::now();
    auto duration=std::chrono::duration_cast<std::chrono::milliseconds>(end-start);
    long local_ms=duration.count();
    long global_ms=0;
    MPI_Reduce(&local_ms, &global_ms, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    if(rank==0) printf("Simulation time: %ld ms\n",global_ms);
    if(printResults && rank==0){
        std::vector<double> bodyData; bodyData.reserve(numBodies*6);
        for(const auto& body:bodies){bodyData.push_back(body.pos.x);bodyData.push_back(body.pos.y);bodyData.push_back(body.pos.z);bodyData.push_back(body.vel.x);bodyData.push_back(body.vel.y);bodyData.push_back(body.vel.z);}
        print_results(bodyData,"Bodies");
    }
    if(validate && rank==0){
        printf("Validating simulation results...\n");
        if(validateSimulation(bodies)){double finalEnergy=computeTotalEnergy(bodies);printf("Final energy: %.6f\n",finalEnergy);printf("Validation: PASSED\n");}
        else {printf("Validation: FAILED\n");MPI_Finalize();return 1;}
    }
    MPI_Finalize();
    return 0;
}
