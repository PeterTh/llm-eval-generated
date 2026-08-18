#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
#include <limits>
#include <vector>
#include "../common/results_output.hpp"

static constexpr double MAX_WIDTH=20.0, MAX_HEIGHT=20.0;
struct Point { double x,y; };
struct Cluster { std::vector<int> members; int seed_point; };

void generateSyntheticData(std::vector<Point>& p, int n, unsigned seed=42) {
  auto rnd=[&seed](){ return rand_r(&seed)/static_cast<double>(RAND_MAX); };
  int count=0; const double md=std::min(MAX_WIDTH,MAX_HEIGHT);
  while(count<n) { double cx=rnd()*MAX_WIDTH, cy=rnd()*MAX_HEIGHT, r=rnd()*md/2;
    int left=std::min(std::max(1,static_cast<int>(rnd()*(n/30.0))),n-count);
    while(left) { double rad=rnd()*r, dx=(2*rnd()-1)*rad;
      double dy=std::sqrt(std::max(0.0,rad*rad-dx*dx))*((rnd()<.5)?-1:1);
      if(cx+dx>=0&&cx+dx<=MAX_WIDTH&&cy+dy>=0&&cy+dy<=MAX_HEIGHT)
        p[count++]={cx+dx,cy+dy},--left;
    }
  }
}
inline double distance(const Point&a,const Point&b){double x=a.x-b.x,y=a.y-b.y;return std::sqrt(x*x+y*y);}

__global__ void candidateDistances(const Point* p,const int* members,int m,const unsigned char* blocked,
                                    int n,double* out) {
  int c=blockIdx.x*blockDim.x+threadIdx.x; if(c>=n||blocked[c]) { if(c<n) out[c]=DBL_MAX; return; }
  double mx=0; for(int i=0;i<m;++i) { double x=p[c].x-p[members[i]].x,y=p[c].y-p[members[i]].y;
    mx=fmax(mx,sqrt(x*x+y*y)); } out[c]=mx;
}

class GpuQT {
  Point* dp=nullptr; int* dm=nullptr; unsigned char* db=nullptr; double* dout=nullptr; int cap=0;
public:
  explicit GpuQT(const std::vector<Point>&p):cap(static_cast<int>(p.size())) {
    cudaMalloc(&dp,p.size()*sizeof(Point)); cudaMemcpy(dp,p.data(),p.size()*sizeof(Point),cudaMemcpyHostToDevice);
    cudaMalloc(&dm,p.size()*sizeof(int)); cudaMalloc(&db,p.size()); cudaMalloc(&dout,p.size()*sizeof(double));
  }
  ~GpuQT(){cudaFree(dp);cudaFree(dm);cudaFree(db);cudaFree(dout);}
  int next(const std::vector<int>& members,const std::vector<unsigned char>& clustered,
           std::vector<double>& vals,double threshold) {
    std::vector<unsigned char> b(cap); for(int i=0;i<cap;++i)b[i]=clustered[i];
    cudaMemcpy(dm,members.data(),members.size()*sizeof(int),cudaMemcpyHostToDevice);
    cudaMemcpy(db,b.data(),b.size(),cudaMemcpyHostToDevice);
    candidateDistances<<<(cap+255)/256,256>>>(dp,dm,members.size(),db,cap,dout);
    cudaMemcpy(vals.data(),dout,cap*sizeof(double),cudaMemcpyDeviceToHost);
    int best=-1; double d=std::numeric_limits<double>::max();
    for(int i=0;i<cap;++i) if(!clustered[i]&&vals[i]<threshold&&vals[i]<d)d=vals[i],best=i;
    return best;
  }
};

std::vector<int> makeCluster(int seed,const std::vector<unsigned char>& clustered,const std::vector<Point>&p,double t,GpuQT&gpu){
  std::vector<int> m{seed}; std::vector<double> vals(p.size());
  while(static_cast<int>(m.size())<static_cast<int>(p.size())) { std::vector<unsigned char> blocked=clustered;
    for(int x:m) blocked[x]=true; int x=gpu.next(m,blocked,vals,t); if(x<0)break; m.push_back(x);
  } return m;
}

std::vector<Cluster> qtClustering(const std::vector<Point>&p,double t,int rank,int size) {
  int n=p.size(); std::vector<unsigned char> clustered(n); std::vector<Cluster> result; GpuQT gpu(p);
  while(true) { int localCard=-1,localSeed=n; std::vector<int> localMembers;
    { std::vector<int> best; int card=-1,seed=n;
      for(int i=rank;i<n;i+=size) if(!clustered[i]) { auto c=makeCluster(i,clustered,p,t,gpu);
        if(static_cast<int>(c.size())>card||(static_cast<int>(c.size())==card&&i<seed))card=c.size(),seed=i,best=std::move(c); }
      localCard=card; localSeed=seed; localMembers=std::move(best);
    }
    std::vector<int> cards(size),seeds(size); MPI_Allgather(&localCard,1,MPI_INT,cards.data(),1,MPI_INT,MPI_COMM_WORLD); MPI_Allgather(&localSeed,1,MPI_INT,seeds.data(),1,MPI_INT,MPI_COMM_WORLD);
    int bestRank=0; for(int r=1;r<size;++r) if(cards[r]>cards[bestRank]||(cards[r]==cards[bestRank]&&seeds[r]<seeds[bestRank]))bestRank=r;
    if(cards[bestRank]<=0||seeds[bestRank]>=n)break; int len=cards[bestRank]; if(rank!=bestRank)localMembers.resize(len);
    MPI_Bcast(localMembers.data(),len,MPI_INT,bestRank,MPI_COMM_WORLD); result.push_back({localMembers,seeds[bestRank]}); for(int x:localMembers)clustered[x]=1;
  } return result;
}

bool validateClusters(const std::vector<Cluster>&c,const std::vector<Point>&p,double t){bool ok=true; std::vector<int> seen(p.size(),-1);
  #pragma omp parallel for schedule(static)
  for(int k=0;k<static_cast<int>(c.size());++k) for(size_t i=0;i<c[k].members.size();++i) for(size_t j=i+1;j<c[k].members.size();++j) if(distance(p[c[k].members[i]],p[c[k].members[j]])>t*1.001) ok=false;
  for(size_t k=0;k<c.size();++k)for(int x:c[k].members){if(seen[x]>=0)ok=false;seen[x]=k;} printf("Total points: %zu, Clustered: %zu, Unclustered: %zu\n",p.size(),std::count_if(seen.begin(),seen.end(),[](int x){return x>=0;}),std::count(seen.begin(),seen.end(),-1)); return ok; }

int main(int argc,char**argv){MPI_Init(&argc,&argv);int rank,size;MPI_Comm_rank(MPI_COMM_WORLD,&rank);MPI_Comm_size(MPI_COMM_WORLD,&size);int n=1000;double t=2;bool val=false,res=false;
  for(int i=1;i<argc;++i)if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]);else if(!strcmp(argv[i],"-t")&&i+1<argc)t=atof(argv[++i]);else if(!strcmp(argv[i],"-v"))val=true;else if(!strcmp(argv[i],"-r"))res=true;else if(!strcmp(argv[i],"-h")){if(!rank)printf("Usage: %s [-n points] [-t threshold] [-v] [-r]\n",argv[0]);MPI_Finalize();return 0;}
  if(n<=0||t<=0){MPI_Finalize();return 1;} std::vector<Point> p(n);generateSyntheticData(p,n); MPI_Barrier(MPI_COMM_WORLD);auto s=std::chrono::high_resolution_clock::now();auto c=qtClustering(p,t,rank,size);auto e=std::chrono::high_resolution_clock::now();
  if(!rank){printf("QT Clustering Benchmark\nNumber of points: %d\nClusters found: %zu\nClustering time: %ld ms\n",n,c.size(),std::chrono::duration_cast<std::chrono::milliseconds>(e-s).count());if(res){std::vector<double>d(n,-1);for(size_t k=0;k<c.size();++k)for(int x:c[k].members)d[x]=k;print_results(d,"ClusterMembership");}if(val)printf("Validation: %s\n",validateClusters(c,p,t)?"PASSED":"FAILED");}MPI_Finalize();return 0;}
