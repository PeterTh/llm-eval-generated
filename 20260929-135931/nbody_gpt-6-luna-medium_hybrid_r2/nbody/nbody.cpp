#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <algorithm>

#include "../common/results_output.hpp"

constexpr double SOFTENING = 1e-9;
constexpr double DT = 0.01;
struct Vec3 { double x, y, z; };
struct Body { Vec3 pos; Vec3 vel; };

#define CUDA_CHECK(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)

__global__ void forceKernel(const double* x, const double* y, const double* z,
                            const double* vx, const double* vy, const double* vz,
                            double* out, int n, int begin, int count) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= count) return;
    int i = begin + k;
    double fx=0.0, fy=0.0, fz=0.0;
    const double xi=x[i], yi=y[i], zi=z[i];
    for (int j=0; j<n; ++j) {
        double dx=x[j]-xi, dy=y[j]-yi, dz=z[j]-zi;
        double inv=rsqrt(dx*dx+dy*dy+dz*dz+SOFTENING);
        double q=inv*inv*inv;
        fx += dx*q; fy += dy*q; fz += dz*q;
    }
    out[6*k+0]=vx[i]+DT*fx; out[6*k+1]=vy[i]+DT*fy; out[6*k+2]=vz[i]+DT*fz;
}

void randomizeBodies(std::vector<Body>& bodies, unsigned int seed=42) {
    for (auto& b:bodies) {
        b.pos.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        b.pos.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        b.pos.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        b.vel.x=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        b.vel.y=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
        b.vel.z=2.0*(rand_r(&seed)/(double)RAND_MAX)-1.0;
    }
}

double computeTotalEnergy(const std::vector<Body>& b) {
    double e=0; size_t n=b.size();
    #pragma omp parallel for reduction(+:e) schedule(static)
    for (long long i=0;i<(long long)n;++i) {
        const auto& a=b[(size_t)i];
        e += 0.5*(a.vel.x*a.vel.x+a.vel.y*a.vel.y+a.vel.z*a.vel.z);
        for (size_t j=(size_t)i+1;j<n;++j) {
            double dx=b[j].pos.x-a.pos.x,dy=b[j].pos.y-a.pos.y,dz=b[j].pos.z-a.pos.z;
            e -= 1.0/std::sqrt(dx*dx+dy*dy+dz*dz+SOFTENING);
        }
    }
    return e;
}

bool validateSimulation(const std::vector<Body>& b) {
    for(const auto& a:b) if(!std::isfinite(a.pos.x)||!std::isfinite(a.pos.y)||!std::isfinite(a.pos.z)||!std::isfinite(a.vel.x)||!std::isfinite(a.vel.y)||!std::isfinite(a.vel.z)||std::abs(a.pos.x)>1e6||std::abs(a.pos.y)>1e6||std::abs(a.pos.z)>1e6||std::abs(a.vel.x)>1e6||std::abs(a.vel.y)>1e6||std::abs(a.vel.z)>1e6) return false;
    return true;
}
void printUsage(const char* p) { printf("Usage: %s [-n num] [-s num] [-v] [-r] [-h]\n",p); }

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0,size=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    int numBodies=1024,numSteps=10; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i) {
        if(!strcmp(argv[i],"-n")&&i+1<argc) numBodies=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-s")&&i+1<argc) numSteps=atoi(argv[++i]);
        else if(!strcmp(argv[i],"-v")) validate=true;
        else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")){if(rank==0)printUsage(argv[0]); MPI_Finalize(); return 0;}
        else {if(rank==0){printf("Unknown option: %s\n",argv[i]);printUsage(argv[0]);} MPI_Finalize();return 1;}
    }
    if(numBodies<0||numSteps<0){if(rank==0)fprintf(stderr,"Body and step counts must be nonnegative\n");MPI_Finalize();return 1;}
    int deviceCount=0; CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if(deviceCount<=0){if(rank==0)fprintf(stderr,"At least one CUDA device is required\n");MPI_Abort(MPI_COMM_WORLD,2);}
    CUDA_CHECK(cudaSetDevice(rank%deviceCount));
    if(rank==0) {printf("N-Body Simulation\nNumber of bodies: %d\nNumber of steps: %d\nValidation: %s\n",numBodies,numSteps,validate?"enabled":"disabled");}
    std::vector<Body> bodies((size_t)numBodies); randomizeBodies(bodies);
    int begin=(int)((long long)numBodies*rank/size), end=(int)((long long)numBodies*(rank+1)/size), local=end-begin;
    std::vector<int> counts(size),displs(size);
    for(int r=0;r<size;++r){int lo=(int)((long long)numBodies*r/size),hi=(int)((long long)numBodies*(r+1)/size);counts[r]=(hi-lo)*6;displs[r]=lo*6;}
    std::vector<double> xs(numBodies),ys(numBodies),zs(numBodies),vxs(numBodies),vys(numBodies),vzs(numBodies),localOut((size_t)local*6),allVel((size_t)numBodies*6);
    double *dx=nullptr,*dy=nullptr,*dz=nullptr,*dvx=nullptr,*dvy=nullptr,*dvz=nullptr,*dout=nullptr;
    CUDA_CHECK(cudaMalloc(&dx,(size_t)numBodies*sizeof(double))); CUDA_CHECK(cudaMalloc(&dy,(size_t)numBodies*sizeof(double))); CUDA_CHECK(cudaMalloc(&dz,(size_t)numBodies*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dvx,(size_t)numBodies*sizeof(double))); CUDA_CHECK(cudaMalloc(&dvy,(size_t)numBodies*sizeof(double))); CUDA_CHECK(cudaMalloc(&dvz,(size_t)numBodies*sizeof(double)));
    CUDA_CHECK(cudaMalloc(&dout,std::max<size_t>(1,(size_t)local*6)*sizeof(double)));
    MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::steady_clock::now();
    for(int step=0;step<numSteps;++step) {
        #pragma omp parallel for schedule(static)
        for(int i=0;i<numBodies;++i){const auto& b=bodies[i];xs[i]=b.pos.x;ys[i]=b.pos.y;zs[i]=b.pos.z;vxs[i]=b.vel.x;vys[i]=b.vel.y;vzs[i]=b.vel.z;}
        CUDA_CHECK(cudaMemcpy(dx,xs.data(),(size_t)numBodies*sizeof(double),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dy,ys.data(),(size_t)numBodies*sizeof(double),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dz,zs.data(),(size_t)numBodies*sizeof(double),cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dvx,vxs.data(),(size_t)numBodies*sizeof(double),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dvy,vys.data(),(size_t)numBodies*sizeof(double),cudaMemcpyHostToDevice)); CUDA_CHECK(cudaMemcpy(dvz,vzs.data(),(size_t)numBodies*sizeof(double),cudaMemcpyHostToDevice));
        if(local>0){int threads=256,blocks=(local+threads-1)/threads;forceKernel<<<blocks,threads>>>(dx,dy,dz,dvx,dvy,dvz,dout,numBodies,begin,local);CUDA_CHECK(cudaGetLastError());CUDA_CHECK(cudaMemcpy(localOut.data(),dout,(size_t)local*6*sizeof(double),cudaMemcpyDeviceToHost));}
        MPI_Allgatherv(localOut.data(),local*6,MPI_DOUBLE,allVel.data(),counts.data(),displs.data(),MPI_DOUBLE,MPI_COMM_WORLD);
        #pragma omp parallel for schedule(static)
        for(int i=0;i<numBodies;++i){auto& b=bodies[i];b.vel.x=allVel[6*i];b.vel.y=allVel[6*i+1];b.vel.z=allVel[6*i+2];b.pos.x+=b.vel.x*DT;b.pos.y+=b.vel.y*DT;b.pos.z+=b.vel.z*DT;}
    }
    MPI_Barrier(MPI_COMM_WORLD); auto finish=std::chrono::steady_clock::now();
    if(rank==0) printf("Simulation time: %lld ms\n",(long long)std::chrono::duration_cast<std::chrono::milliseconds>(finish-start).count());
    if(rank==0&&printResults){std::vector<double> data;data.reserve((size_t)numBodies*6);for(const auto& b:bodies){data.insert(data.end(),{b.pos.x,b.pos.y,b.pos.z,b.vel.x,b.vel.y,b.vel.z});}print_results(data,"Bodies");}
    int valid=1; if(rank==0&&validate){printf("Validating simulation results...\n");valid=validateSimulation(bodies);if(valid)printf("Final energy: %.6f\n",computeTotalEnergy(bodies));printf("Validation: %s\n",valid?"PASSED":"FAILED");}
    MPI_Bcast(&valid,1,MPI_INT,0,MPI_COMM_WORLD);
    CUDA_CHECK(cudaFree(dx));CUDA_CHECK(cudaFree(dy));CUDA_CHECK(cudaFree(dz));CUDA_CHECK(cudaFree(dvx));CUDA_CHECK(cudaFree(dvy));CUDA_CHECK(cudaFree(dvz));CUDA_CHECK(cudaFree(dout));
    MPI_Finalize();return validate&&!valid?1:0;
}
