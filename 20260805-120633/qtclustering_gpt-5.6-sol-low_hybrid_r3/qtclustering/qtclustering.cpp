// Hybrid MPI + OpenMP + CUDA QT clustering benchmark.
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0, MAX_HEIGHT = 20.0;
struct Point { double x, y; };
struct Cluster { std::vector<int> members; int seed_point; };

#define CUDA_OK(call) do { cudaError_t e_ = (call); if (e_ != cudaSuccess) { \
  fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e_)); MPI_Abort(MPI_COMM_WORLD, 2); } } while (0)

void generateSyntheticData(std::vector<Point>& p, int n, unsigned seed=42) {
  auto frand=[&seed](){return rand_r(&seed)/static_cast<double>(RAND_MAX);};
  int count=0;
  while(count<n) {
    double cx=frand()*MAX_WIDTH, cy=frand()*MAX_HEIGHT, rmax=frand()*10.0;
    int cnt=static_cast<int>(frand()*(n/30.0));
    cnt=std::min(cnt,n-count);
    while(cnt>0) {
      double sign=frand()<.5?-1.:1., r=frand()*rmax;
      double dx=(2.*frand()-1.)*r, dy=sqrt(r*r-dx*dx)*sign;
      double x=cx+dx,y=cy+dy;
      if(x<0||x>MAX_WIDTH||y<0||y>MAX_HEIGHT) continue;
      p[count++]={x,y}; --cnt;
    }
  }
}

__device__ __forceinline__ bool better(double da,int ia,double db,int ib) {
  return ia>=0 && (ib<0 || da<db || (da==db && ia<ib));
}

// A block constructs one candidate cluster. Candidates are split over threads;
// every thread evaluates its points serially against the current member list.
__global__ void candidateKernel(const Point* __restrict__ points,
                                const unsigned char* __restrict__ clustered,
                                const int* __restrict__ seeds, int seed_count, int n,
                                double threshold, unsigned char* states,
                                int* members, int* sizes) {
  int b=blockIdx.x;
  if(b>=seed_count) return;
  unsigned char* in=states+static_cast<size_t>(b)*n;
  int* out=members+static_cast<size_t>(b)*n;
  for(int i=threadIdx.x;i<n;i+=blockDim.x) in[i]=0;
  __syncthreads();
  int seed=seeds[b];
  if(threadIdx.x==0) { in[seed]=1; out[0]=seed; sizes[b]=1; }
  __syncthreads();
  extern __shared__ unsigned char smem[];
  double* sd=reinterpret_cast<double*>(smem);
  int* si=reinterpret_cast<int*>(sd+blockDim.x);
  for(int count=1;count<n;++count) {
    double best=DBL_MAX; int bi=-1;
    for(int c=threadIdx.x;c<n;c+=blockDim.x) {
      if(clustered[c]||in[c]) continue;
      double md=0.;
      #pragma unroll 1
      for(int j=0;j<count;++j) {
        double dx=points[c].x-points[out[j]].x, dy=points[c].y-points[out[j]].y;
        double d=sqrt(dx*dx+dy*dy); md=d>md?d:md;
      }
      if(md<threshold && better(md,c,best,bi)) { best=md; bi=c; }
    }
    sd[threadIdx.x]=best; si[threadIdx.x]=bi; __syncthreads();
    for(int stride=blockDim.x/2;stride;stride>>=1) {
      if(threadIdx.x<stride && better(sd[threadIdx.x+stride],si[threadIdx.x+stride],sd[threadIdx.x],si[threadIdx.x])) {
        sd[threadIdx.x]=sd[threadIdx.x+stride]; si[threadIdx.x]=si[threadIdx.x+stride];
      }
      __syncthreads();
    }
    int chosen=si[0];
    if(chosen<0) break;
    if(threadIdx.x==0) { in[chosen]=1; out[count]=chosen; sizes[b]=count+1; }
    __syncthreads();
  }
}

static std::vector<Cluster> qtClustering(const std::vector<Point>& points,int rank,int ranks,double threshold) {
  int n=static_cast<int>(points.size());
  std::vector<unsigned char> clustered(n,0); std::vector<Cluster> result;
  Point* dp=nullptr; unsigned char* dc=nullptr;
  CUDA_OK(cudaMalloc(&dp,sizeof(Point)*n)); CUDA_OK(cudaMalloc(&dc,n));
  CUDA_OK(cudaMemcpy(dp,points.data(),sizeof(Point)*n,cudaMemcpyHostToDevice));
  size_t free_mem,total_mem; CUDA_OK(cudaMemGetInfo(&free_mem,&total_mem));
  // State and member arrays dominate memory. Batching keeps a substantial safety margin.
  size_t per_seed=static_cast<size_t>(n)*(sizeof(int)+sizeof(unsigned char));
  int batch_cap=static_cast<int>(std::max<size_t>(1,std::min<size_t>(n,(free_mem*3/5)/std::max<size_t>(1,per_seed))));
  int *ds=nullptr,*dm=nullptr,*dz=nullptr; unsigned char* di=nullptr;
  CUDA_OK(cudaMalloc(&ds,sizeof(int)*batch_cap)); CUDA_OK(cudaMalloc(&dz,sizeof(int)*batch_cap));
  CUDA_OK(cudaMalloc(&dm,sizeof(int)*static_cast<size_t>(batch_cap)*n));
  CUDA_OK(cudaMalloc(&di,static_cast<size_t>(batch_cap)*n));
  std::vector<int> seeds,hsizes(batch_cap),best_members(n); seeds.reserve((n+ranks-1)/ranks);
  while(true) {
    seeds.clear();
    for(int i=rank;i<n;i+=ranks) if(!clustered[i]) seeds.push_back(i);
    int local_card=-1,local_seed=std::numeric_limits<int>::max();
    for(size_t off=0;off<seeds.size();off+=batch_cap) {
      int b=static_cast<int>(std::min<size_t>(batch_cap,seeds.size()-off));
      CUDA_OK(cudaMemcpy(ds,seeds.data()+off,sizeof(int)*b,cudaMemcpyHostToDevice));
      CUDA_OK(cudaMemcpy(dc,clustered.data(),n,cudaMemcpyHostToDevice));
      candidateKernel<<<b,256,256*(sizeof(double)+sizeof(int))>>>(dp,dc,ds,b,n,threshold,di,dm,dz);
      CUDA_OK(cudaGetLastError()); CUDA_OK(cudaMemcpy(hsizes.data(),dz,sizeof(int)*b,cudaMemcpyDeviceToHost));
      for(int j=0;j<b;++j) if(hsizes[j]>local_card || (hsizes[j]==local_card && seeds[off+j]<local_seed)) {
        local_card=hsizes[j]; local_seed=seeds[off+j];
        CUDA_OK(cudaMemcpy(best_members.data(),dm+static_cast<size_t>(j)*n,sizeof(int)*local_card,cudaMemcpyDeviceToHost));
      }
    }
    struct { int card; int seed; } in={local_card,local_seed},out;
    MPI_Allreduce(&in,&out,1,MPI_2INT,MPI_MAXLOC,MPI_COMM_WORLD);
    if(out.card<=0) break;
    int winner=out.seed, owner=winner%ranks;
    std::vector<int> chosen(out.card);
    if(rank==owner) std::copy_n(best_members.begin(),out.card,chosen.begin());
    MPI_Bcast(chosen.data(),out.card,MPI_INT,owner,MPI_COMM_WORLD);
    #pragma omp parallel for schedule(static)
    for(int j=0;j<out.card;++j) clustered[chosen[j]]=1;
    result.push_back({std::move(chosen),winner});
  }
  cudaFree(di); cudaFree(dm); cudaFree(dz); cudaFree(ds); cudaFree(dc); cudaFree(dp);
  return result;
}

static bool validateClusters(const std::vector<Cluster>& cs,const std::vector<Point>& p,double threshold) {
  int valid=1; std::vector<double> diameters(cs.size()); printf("Validating clusters:\n");
  #pragma omp parallel for schedule(dynamic) reduction(&:valid)
  for(long c=0;c<(long)cs.size();++c) {
    double md=0.;
    for(size_t i=0;i<cs[c].members.size();++i) for(size_t j=i+1;j<cs[c].members.size();++j) {
      Point a=p[cs[c].members[i]],b=p[cs[c].members[j]]; double dx=a.x-b.x,dy=a.y-b.y;
      md=std::max(md,sqrt(dx*dx+dy*dy));
    }
    diameters[c]=md;
    if(md>threshold*1.001) valid=0;
  }
  for(size_t c=0;c<std::min<size_t>(10,cs.size());++c)
    printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n",c,cs[c].members.size(),cs[c].seed_point,diameters[c]);
  std::vector<int> membership(p.size(),-1); int count=0;
  for(size_t c=0;c<cs.size();++c) for(int x:cs[c].members) { if(membership[x]>=0) valid=0; else {membership[x]=c;++count;} }
  printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",p.size(),count,p.size()-count);
  return valid;
}

static void usage(const char* p) { printf("Usage: %s [-n points] [-t threshold] [-v] [-r] [-h]\n",p); }
int main(int argc,char** argv) {
  int provided=0; MPI_Init_thread(&argc,&argv,MPI_THREAD_FUNNELED,&provided); int rank,ranks; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&ranks);
  int devices=0; CUDA_OK(cudaGetDeviceCount(&devices)); if(!devices) { if(!rank) fprintf(stderr,"CUDA device required\n"); MPI_Abort(MPI_COMM_WORLD,2); }
  int local_rank=rank; MPI_Comm local; MPI_Comm_split_type(MPI_COMM_WORLD,MPI_COMM_TYPE_SHARED,rank,MPI_INFO_NULL,&local);
  MPI_Comm_rank(local,&local_rank); MPI_Comm_free(&local); CUDA_OK(cudaSetDevice(local_rank%devices));
  int n=1000; double threshold=2.; bool validate=false,prints=false; int parse_ok=1,help=0;
  for(int i=1;i<argc;++i) if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]); else if(!strcmp(argv[i],"-t")&&i+1<argc)threshold=atof(argv[++i]); else if(!strcmp(argv[i],"-v"))validate=true; else if(!strcmp(argv[i],"-r"))prints=true; else if(!strcmp(argv[i],"-h"))help=1; else parse_ok=0;
  if(help||!parse_ok||n<=0||threshold<=0) { if(!rank) usage(argv[0]); MPI_Finalize(); return help?0:1; }
  std::vector<Point> points(n); if(!rank) generateSyntheticData(points,n);
  MPI_Bcast(points.data(),n*sizeof(Point),MPI_BYTE,0,MPI_COMM_WORLD);
  if(!rank) printf("QT Clustering Benchmark\nNumber of points: %d\nDistance threshold: %.2f\nValidation: %s\nMPI ranks: %d, OpenMP threads/rank: %d\n",n,threshold,validate?"enabled":"disabled",ranks,omp_get_max_threads());
  MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime(); auto cs=qtClustering(points,rank,ranks,threshold); double local_sec=MPI_Wtime()-start,sec=0.; MPI_Reduce(&local_sec,&sec,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
  int rc=0;
  if(!rank) {
    int total=0,mx=0; for(auto& c:cs){total+=c.members.size();mx=std::max(mx,(int)c.members.size());}
    printf("Clustering time: %ld ms\nClusters found: %zu\nPoints clustered: %d / %d (%.1f%%)\nAverage cluster size: %.2f\nMaximum cluster size: %d\nPerformance: %.1f clusters/s, %.1f points/s\n",(long)(sec*1000),cs.size(),total,n,100.*total/n,cs.empty()?0.:double(total)/cs.size(),mx,cs.size()/sec,n/sec);
    if(prints) { std::vector<double> m(n,-1); for(size_t c=0;c<cs.size();++c) for(int x:cs[c].members)m[x]=c; print_results(m,"ClusterMembership"); }
    if(validate) { bool ok=validateClusters(cs,points,threshold); printf("Validation: %s\n",ok?"PASSED":"FAILED"); rc=ok?0:1; }
  }
  MPI_Bcast(&rc,1,MPI_INT,0,MPI_COMM_WORLD); MPI_Finalize(); return rc;
}
