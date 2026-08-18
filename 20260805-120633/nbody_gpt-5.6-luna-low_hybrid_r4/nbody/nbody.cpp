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
struct Vec3 { double x,y,z; };
struct Body { Vec3 pos, vel; };

__global__ void step_kernel(Body* local, const Body* all, int first, int localN, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= localN) return;
    Body b = all[first + i];
    double fx=0.0, fy=0.0, fz=0.0;
    for (int j=0; j<n; ++j) {
        double dx=all[j].pos.x-b.pos.x, dy=all[j].pos.y-b.pos.y, dz=all[j].pos.z-b.pos.z;
        double inv=1.0/sqrt(dx*dx+dy*dy+dz*dz+SOFTENING), inv3=inv*inv*inv;
        fx += dx*inv3; fy += dy*inv3; fz += dz*inv3;
    }
    b.vel.x += DT*fx; b.vel.y += DT*fy; b.vel.z += DT*fz;
    b.pos.x += b.vel.x*DT; b.pos.y += b.vel.y*DT; b.pos.z += b.vel.z*DT;
    local[i]=b;
}

static void die_cuda(cudaError_t e, const char* where) {
    if (e != cudaSuccess) { fprintf(stderr,"CUDA error at %s: %s\n",where,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1); }
}
void randomizeBodies(std::vector<Body>& b) { unsigned seed=42; for (auto& x:b) {
    x.pos.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; x.pos.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; x.pos.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
    x.vel.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; x.vel.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; x.vel.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0; }}
bool validateSimulation(const std::vector<Body>& b) { bool ok=true;
#pragma omp parallel for reduction(&:ok)
    for (long i=0;i<(long)b.size();++i) { const Body& x=b[i]; for(double v:{x.pos.x,x.pos.y,x.pos.z,x.vel.x,x.vel.y,x.vel.z}) if(!std::isfinite(v)||std::abs(v)>1e6) ok=false; } return ok; }
double energy(const std::vector<Body>& b) { double e=0; long n=b.size();
#pragma omp parallel for reduction(+:e)
    for(long i=0;i<n;++i) { e+=.5*(b[i].vel.x*b[i].vel.x+b[i].vel.y*b[i].vel.y+b[i].vel.z*b[i].vel.z); for(long j=i+1;j<n;++j) { double x=b[j].pos.x-b[i].pos.x,y=b[j].pos.y-b[i].pos.y,z=b[j].pos.z-b[i].pos.z; e-=1.0/sqrt(x*x+y*y+z*z+SOFTENING); }} return e; }
void usage(const char* p){printf("Usage: %s [-n num] [-s steps] [-v] [-r] [-h]\n",p);}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    int n=1024, steps=10; bool val=false, results=false;
    for(int i=1;i<argc;++i) { if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]); else if(!strcmp(argv[i],"-s")&&i+1<argc)steps=atoi(argv[++i]); else if(!strcmp(argv[i],"-v"))val=true; else if(!strcmp(argv[i],"-r"))results=true; else if(!strcmp(argv[i],"-h")){if(!rank)usage(argv[0]); MPI_Finalize(); return 0;} else {if(!rank)usage(argv[0]); MPI_Finalize(); return 1;} }
    if(n<0||steps<0){MPI_Finalize();return 1;} std::vector<Body> all(n); if(!rank)randomizeBodies(all);
    std::vector<int> bodyCounts(size), bodyDispls(size), counts(size), displs(size);
    for(int r=0;r<size;++r){int a=n*r/size,b=n*(r+1)/size;bodyCounts[r]=b-a;bodyDispls[r]=a;counts[r]=bodyCounts[r]*sizeof(Body);displs[r]=a*sizeof(Body);}
    std::vector<Body> local(bodyCounts[rank]); MPI_Bcast(all.data(),n*sizeof(Body),MPI_BYTE,0,MPI_COMM_WORLD);
    Body *da=nullptr,*dl=nullptr; die_cuda(cudaMalloc(&da,n*sizeof(Body)),"cudaMalloc(all)"); die_cuda(cudaMalloc(&dl,local.size()*sizeof(Body)),"cudaMalloc(local)");
    die_cuda(cudaMemcpy(da,all.data(),n*sizeof(Body),cudaMemcpyHostToDevice),"copy"); cudaEvent_t start,finish; cudaEventCreate(&start); cudaEventCreate(&finish); cudaEventRecord(start);
    for(int s=0;s<steps;++s){ die_cuda(cudaMemcpy(da,all.data(),n*sizeof(Body),cudaMemcpyHostToDevice),"copy");
        if (bodyCounts[rank] > 0) { step_kernel<<<(bodyCounts[rank]+255)/256,256>>>(dl,da,bodyDispls[rank],bodyCounts[rank],n); die_cuda(cudaGetLastError(),"kernel"); }
        die_cuda(cudaMemcpy(local.data(),dl,local.size()*sizeof(Body),cudaMemcpyDeviceToHost),"copy back");
        MPI_Allgatherv(local.data(),counts[rank],MPI_BYTE,all.data(),counts.data(),displs.data(),MPI_BYTE,MPI_COMM_WORLD);
    }
    cudaEventRecord(finish); cudaEventSynchronize(finish); float ms=0; cudaEventElapsedTime(&ms,start,finish); cudaFree(da); cudaFree(dl);
    if(!rank){printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\nSimulation time: %.0f ms\n",n,steps,val?"enabled":"disabled",ms); if(results){std::vector<double>d;d.reserve(n*6);for(auto&x:all){d.insert(d.end(),{x.pos.x,x.pos.y,x.pos.z,x.vel.x,x.vel.y,x.vel.z});}print_results(d,"Bodies");} if(val){printf("Final energy: %.6f\nValidation: %s\n",energy(all),validateSimulation(all)?"PASSED":"FAILED");}}
    MPI_Finalize(); return 0;
}
