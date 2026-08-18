#include <mpi.h>
#include <cuda_runtime.h>
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "../common/results_output.hpp"

using val_t=double; using idx_t=uint64_t;
struct HostState { std::vector<val_t> e, f; int first=0, rows=0, n=0, local=0; };

__global__ void updateKernel(const val_t* in, val_t* out, val_t* flux,
                             const unsigned char* mat, int first, int rows, int n, int ld, int it) {
  int q=blockIdx.x*blockDim.x+threadIdx.x, count=rows*n; if(q>=count) return;
  int r=q/n, c=q-r*n, gr=first+r; val_t ext=(mat[q]? (mat[q]==1?0.5:-0.5):0.0), t=ext;
  val_t self=in[(r+1)*ld+c];
  if(gr>0) t+=(in[r*ld+c]-self)*0.8*0.25;
  if(gr+1<n) t+=(in[(r+2)*ld+c]-self)*0.8*0.25;
  if(c>0) t+=(in[(r+1)*ld+c-1]-self)*0.8*0.25;
  if(c+1<n) t+=(in[(r+1)*ld+c+1]-self)*0.8*0.25;
  out[(r+1)*ld+c]=self+t; flux[q]=t; // total flux is accumulated below on the host/device
  if(it) flux[q]=flux[q];
}

static void check(cudaError_t e,const char* s){if(e!=cudaSuccess){fprintf(stderr,"CUDA %s: %s\n",s,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,2);}}
static void split(int n,int p,int rank,int& first,int& rows){int base=n/p, rem=n%p; rows=base+(rank<rem); first=rank*base+std::min(rank,rem);}

static uint64_t hashResult(const std::vector<val_t>& e,const std::vector<val_t>& f){uint64_t h=0; for(size_t i=0;i<e.size();++i){uint64_t a=*reinterpret_cast<const uint64_t*>(&e[i]),b=*reinterpret_cast<const uint64_t*>(&f[i]);h^=(a+i)*0x9e3779b97f4a7c15ULL;h^=(b+i)*0xbf58476d1ce4e5b9ULL;}return h;}
static void usage(const char* p){printf("Usage: %s [-n num] [-i num] [-v] [-r] [-h]\n",p);}

int main(int argc,char** argv){
 MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
 int n=512,iters=10; bool validate=false, print=false;
 for(int i=1;i<argc;i++){if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]);else if(!strcmp(argv[i],"-i")&&i+1<argc)iters=atoi(argv[++i]);else if(!strcmp(argv[i],"-v"))validate=true;else if(!strcmp(argv[i],"-r"))print=true;else if(!strcmp(argv[i],"-h")){if(!rank)usage(argv[0]);MPI_Finalize();return 0;}else{if(!rank)usage(argv[0]);MPI_Finalize();return 1;}}
 if(n<1||iters<0){if(!rank)fprintf(stderr,"invalid dimensions\n");MPI_Finalize();return 1;}
 int first,rows; split(n,size,rank,first,rows); int ld=n, owned=rows*n;
 cudaSetDevice(rank%std::max(1,[](){int x=0;cudaGetDeviceCount(&x);return x;}()));
 HostState s; s.n=n;s.first=first;s.rows=rows;s.local=(rows+2)*n;s.e.assign((rows+2)*n,0);s.f.assign(owned,0); std::vector<unsigned char> mat(owned,0);
 #pragma omp parallel for
 for(int r=0;r<rows;r++)for(int c=0;c<n;c++){int g=(first+r)*n+c;mat[r*n+c]=(g==0||g==n*n-1)?1:((g==n-1||g==(n-1)*n)?2:0);}
 val_t *di,*do_,*df; unsigned char* dm; size_t all=(rows+2)*(size_t)n; check(cudaMalloc(&di,all*sizeof(val_t)),"alloc");check(cudaMalloc(&do_,all*sizeof(val_t)),"alloc");check(cudaMalloc(&df,owned*sizeof(val_t)),"alloc");check(cudaMalloc(&dm,owned),"alloc");
 check(cudaMemcpy(di,s.e.data(),all*sizeof(val_t),cudaMemcpyHostToDevice),"copy");check(cudaMemcpy(dm,mat.data(),owned,cudaMemcpyHostToDevice),"copy");
 std::vector<val_t> send(n),recv(n),delta(owned); MPI_Barrier(MPI_COMM_WORLD); auto start=std::chrono::high_resolution_clock::now();
 for(int k=0;k<iters;k++){
  if(rank){
   #pragma omp parallel for
   for(int c=0;c<n;c++)send[c]=s.e[n+c];
   MPI_Sendrecv(send.data(),n,MPI_DOUBLE,rank-1,7,recv.data(),n,MPI_DOUBLE,rank-1,8,MPI_COMM_WORLD,MPI_STATUS_IGNORE);std::copy(recv.begin(),recv.end(),s.e.begin());
  }
  if(rank<size-1){
   #pragma omp parallel for
   for(int c=0;c<n;c++)send[c]=s.e[rows*n+c];
   MPI_Sendrecv(send.data(),n,MPI_DOUBLE,rank+1,8,recv.data(),n,MPI_DOUBLE,rank+1,7,MPI_COMM_WORLD,MPI_STATUS_IGNORE);std::copy(recv.begin(),recv.end(),s.e.begin()+(rows+1)*n);
  }
  check(cudaMemcpy(di,s.e.data(),all*sizeof(val_t),cudaMemcpyHostToDevice),"halo"); updateKernel<<<(owned+255)/256,256>>>(di,do_,df,dm,first,rows,n,ld,k);check(cudaGetLastError(),"kernel");check(cudaMemcpy(s.e.data(),do_,all*sizeof(val_t),cudaMemcpyDeviceToHost),"result");
  check(cudaMemcpy(delta.data(),df,owned*sizeof(val_t),cudaMemcpyDeviceToHost),"flux");
  #pragma omp parallel for
  for(int q=0;q<owned;q++)s.f[q]+=std::abs(delta[q]);
 }
 auto end=std::chrono::high_resolution_clock::now(); std::vector<val_t> ge, gf; if(!rank){ge.resize((size_t)n*n);gf.resize((size_t)n*n);} std::vector<int> counts(size),displs(size);for(int r=0;r<size;r++){int f,rr;split(n,size,r,f,rr);counts[r]=rr*n;displs[r]=f*n;}
 MPI_Gatherv(s.e.data()+n,owned,MPI_DOUBLE,ge.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);MPI_Gatherv(s.f.data(),owned,MPI_DOUBLE,gf.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
 if(!rank){long ms=std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();printf("Unstructured Mesh Energy Transfer Benchmark\nGrid size: %d x %d = %d elements\nIterations: %d\nComputation time: %ld ms\n",n,n,n*n,iters,ms);double es=0,fs=0,mn=std::numeric_limits<double>::max(),mx=std::numeric_limits<double>::lowest();for(size_t i=0;i<ge.size();i++){es+=ge[i];fs+=gf[i];mn=std::min(mn,ge[i]);mx=std::max(mx,ge[i]);}printf("Performance:\n  Result hash: %016lX\n",hashResult(ge,gf));if(print)print_results(ge,"ElementEnergy");if(validate)printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n  Validation: %s\n",es,fs,mn,mx,(std::isfinite(es)&&std::isfinite(fs)&&std::isfinite(mn)&&std::isfinite(mx))?"PASSED":"FAILED");}
 cudaFree(di);cudaFree(do_);cudaFree(df);cudaFree(dm);MPI_Finalize();return 0;
}
