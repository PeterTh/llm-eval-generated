#include <algorithm>
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

using Real = double;
inline constexpr size_t idx3(size_t x,size_t y,size_t z,size_t nx,size_t ny) noexcept { return z*(nx*ny)+y*nx+x; }

#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess){fprintf(stderr,"CUDA error: %s\n",cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2);} } while(0)

__global__ void stencilKernel(const Real* in, Real* out, int nx,int ny,int localNz,int globalZ0,int globalNz) {
    size_t q=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    size_t n=(size_t)nx*ny*(localNz-2);
    if(q>=n) return;
    int x=q%nx, y=(q/nx)%ny, lz=(int)(q/((size_t)nx*ny))+1;
    int gz=globalZ0+lz;
    size_t i=idx3(x,y,lz,nx,ny);
    if(x==0||x==nx-1||y==0||y==ny-1||gz==0||gz==globalNz-1) out[i]=in[i];
    else out[i]=(in[i]+in[i-1]+in[i+1]+in[i-nx]+in[i+nx]+in[i-(size_t)nx*ny]+in[i+(size_t)nx*ny])/7.0;
}

void printUsage(const char* p){printf("Usage: %s [-x n] [-y n] [-z n] [-i n] [-v] [-r] [-h]\n",p);}

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv);
    int rank=0,ranks=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
    int localDeviceCount=0; CUDA(cudaGetDeviceCount(&localDeviceCount));
    if(localDeviceCount<=0){if(rank==0)fprintf(stderr,"No CUDA device available\n"); MPI_Abort(MPI_COMM_WORLD,2);}
    int localRank=0; MPI_Comm localComm; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&localComm); MPI_Comm_rank(localComm,&localRank); CUDA(cudaSetDevice(localRank%localDeviceCount)); MPI_Comm_free(&localComm);
    size_t nx=128,ny=0,nz=0; int iterations=10; bool validate=false,printResults=false;
    for(int i=1;i<argc;++i){ if(!strcmp(argv[i],"-x")&&i+1<argc) nx=atoi(argv[++i]); else if(!strcmp(argv[i],"-y")&&i+1<argc) ny=atoi(argv[++i]); else if(!strcmp(argv[i],"-z")&&i+1<argc) nz=atoi(argv[++i]); else if(!strcmp(argv[i],"-i")&&i+1<argc) iterations=atoi(argv[++i]); else if(!strcmp(argv[i],"-v")) validate=true; else if(!strcmp(argv[i],"-r")) printResults=true; else if(!strcmp(argv[i],"-h")){if(rank==0)printUsage(argv[0]); MPI_Finalize(); return 0;} else {if(rank==0){printf("Unknown option: %s\n",argv[i]);printUsage(argv[0]);} MPI_Finalize();return 1;} }
    if(!ny)ny=nx;if(!nz)nz=nx;
    if(nx<3||ny<3||nz<3||ranks>(int)(nz-2)){if(rank==0)fprintf(stderr,"Dimensions must be >=3 and MPI ranks <= nz-2\n");MPI_Abort(MPI_COMM_WORLD,1);}
    size_t inner=nz-2, base=inner/ranks, rem=inner%ranks;
    size_t localInner=base+(rank<(int)rem), globalStart=1+(size_t)rank*base+std::min((size_t)rank,rem), localNz=localInner+2;
    size_t plane=nx*ny, count=plane*localNz;
    std::vector<Real> a(count),b(count);
    #pragma omp parallel for schedule(static)
    for(long long k=0;k<(long long)count;++k){size_t gz=globalStart+(size_t)k-1; a[(size_t)k]=(idx3((size_t)k%nx,((size_t)k/nx)%ny,gz,nx,ny)%19)*1.0;}
    if(rank==0){printf("3D Stencil Benchmark\nGrid size: %zu x %zu x %zu\nIterations: %d\nValidation: %s\n",nx,ny,nz,iterations,validate?"enabled":"disabled");}
    Real *da=nullptr,*db=nullptr; CUDA(cudaMalloc(&da,count*sizeof(Real))); CUDA(cudaMalloc(&db,count*sizeof(Real)));
    CUDA(cudaMemcpy(da,a.data(),count*sizeof(Real),cudaMemcpyHostToDevice)); CUDA(cudaMemcpy(db,b.data(),count*sizeof(Real),cudaMemcpyHostToDevice));
    int prev=rank?rank-1:MPI_PROC_NULL,next=rank+1<ranks?rank+1:MPI_PROC_NULL;
    auto start=std::chrono::high_resolution_clock::now();
    for(int it=0;it<iterations;++it){
        Real *in=(it%2==0)?da:db,*out=(it%2==0)?db:da;
        CUDA(cudaMemcpy(a.data()+plane,in+plane,plane*sizeof(Real),cudaMemcpyDeviceToHost));
        CUDA(cudaMemcpy(a.data()+(localNz-2)*plane,in+(localNz-2)*plane,plane*sizeof(Real),cudaMemcpyDeviceToHost));
        MPI_Sendrecv(a.data()+plane, (int)plane, MPI_DOUBLE,prev,10,a.data()+(localNz-1)*plane,(int)plane,MPI_DOUBLE,next,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(a.data()+(localNz-2)*plane,(int)plane,MPI_DOUBLE,next,11,a.data(),(int)plane, MPI_DOUBLE,prev,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        if(prev!=MPI_PROC_NULL)CUDA(cudaMemcpy(in,a.data(),plane*sizeof(Real),cudaMemcpyHostToDevice));
        if(next!=MPI_PROC_NULL)CUDA(cudaMemcpy(in+(localNz-1)*plane,a.data()+(localNz-1)*plane,plane*sizeof(Real),cudaMemcpyHostToDevice));
        size_t work=nx*ny*localInner; int threads=256; int blocks=(int)((work+threads-1)/threads);
        stencilKernel<<<blocks,threads>>>(in,out,(int)nx,(int)ny,(int)localNz,(int)globalStart-1,(int)nz); CUDA(cudaGetLastError());
    }
    Real *finalD=(iterations%2==0)?da:db; CUDA(cudaMemcpy(a.data()+plane,finalD+plane,localInner*plane*sizeof(Real),cudaMemcpyDeviceToHost));
    auto end=std::chrono::high_resolution_clock::now();
    // Gather each rank's owned z planes; rank-local halos are excluded.
    std::vector<int> counts(ranks),displs(ranks); for(int r=0;r<ranks;++r){size_t ni=base+(r<(int)rem); counts[r]=(int)(ni*plane); size_t gs=1+(size_t)r*base+std::min((size_t)r,rem); displs[r]=(int)(gs*plane);}
    std::vector<Real> full; if(rank==0){full.resize(nx*ny*nz);
        #pragma omp parallel for schedule(static)
        for(long long k=0;k<(long long)full.size();++k)full[(size_t)k]=(size_t)k%19;}
    MPI_Gatherv(a.data()+plane,(int)(localInner*plane),MPI_DOUBLE,rank==0?full.data():nullptr,counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    long long ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count(); long long maxms=0; MPI_Reduce(&ms,&maxms,1,MPI_LONG_LONG_INT,MPI_MAX,0,MPI_COMM_WORLD);
    if(rank==0){printf("Computation time: %lld ms\n",maxms);double updates=(double)((nx-2)*(ny-2)*(nz-2))*iterations;printf("Performance: %.3f MCellUpdates/s\n",updates/(maxms/1000.0)/1e6);if(printResults)print_results(full,"Grid");if(validate){bool ok=true;
        #pragma omp parallel for reduction(&:ok)
        for(long long i=0;i<(long long)full.size();++i)if(!std::isfinite(full[(size_t)i])||std::abs(full[(size_t)i])>1e6)ok=false;
        auto mm=std::minmax_element(full.begin(),full.end());printf("Value range: [%.6f, %.6f]\n",*mm.first,*mm.second);printf("Validation: %s\n",ok?"PASSED":"FAILED");if(!ok)ms=-1;}}
    MPI_Bcast(&ms,1,MPI_LONG_LONG_INT,0,MPI_COMM_WORLD); CUDA(cudaFree(da));CUDA(cudaFree(db));MPI_Finalize();return ms<0?1:0;
}
