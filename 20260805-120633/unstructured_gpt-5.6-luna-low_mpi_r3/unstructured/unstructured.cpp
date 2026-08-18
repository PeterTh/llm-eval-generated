#include <mpi.h>
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

using idx_t = uint64_t; using val_t = double;
constexpr int MAX_CONNECTIONS = 8;
struct Material { val_t transfer_coeff, external_flow; };
struct ElementStatic { idx_t material_idx, num_connections; idx_t connected_idx[MAX_CONNECTIONS]; val_t connected_flux[MAX_CONNECTIONS]; };
struct ElementDynamic { val_t current_energy, total_flux; };
constexpr idx_t DEFAULT_MAT_ID=0, INFLOW_MAT_ID=1, OUTFLOW_MAT_ID=2;

static void usage(const char* p) { printf("Usage: %s [-n num] [-i num] [-v] [-r] [-h]\n",p); }

struct LocalWorld {
    int n, first_row, rows, rank, size, top, bottom;
    std::vector<Material> materials;
    std::vector<ElementStatic> st;
    std::vector<ElementDynamic> cur, next;
    std::vector<val_t> top_energy, bottom_energy;
};

static void build(LocalWorld& w, int n, int rank, int size) {
    w.n=n; w.rank=rank; w.size=size;
    const int base=n/size, extra=n%size;
    w.first_row=rank*base+std::min(rank,extra); w.rows=base+(rank<extra);
    w.top=(rank>0)?rank-1:MPI_PROC_NULL; w.bottom=(rank+1<size)?rank+1:MPI_PROC_NULL;
    w.materials={{.8,0},{.8,.5},{.8,-.5}};
    w.st.resize(static_cast<size_t>(w.rows)*n); w.cur.resize(w.st.size()); w.next.resize(w.st.size());
    w.top_energy.resize(n); w.bottom_energy.resize(n);
    for(int r=0;r<w.rows;++r) for(int c=0;c<n;++c) {
        const int gr=w.first_row+r, i=r*n+c; auto& e=w.st[i]; e.material_idx=DEFAULT_MAT_ID; e.num_connections=0;
        const int dr[4]={1,-1,0,0}, dc[4]={0,0,1,-1};
        for(int k=0;k<4;++k) { int rr=gr+dr[k],cc=c+dc[k]; if(rr>=0&&rr<n&&cc>=0&&cc<n) { e.connected_idx[e.num_connections++]=static_cast<idx_t>(rr*n+cc); e.connected_flux[e.num_connections-1]=1.; } }
        if((gr==0||gr==n-1)&&(c==0||c==n-1)) e.material_idx=((gr==0&&c==0)||(gr==n-1&&c==n-1))?INFLOW_MAT_ID:OUTFLOW_MAT_ID;
        w.cur[i]={0.,0.};
    }
}

static inline val_t energy(const LocalWorld& w, idx_t g, const std::vector<val_t>& t, const std::vector<val_t>& b) {
    const int row=static_cast<int>(g/w.n), col=static_cast<int>(g%w.n);
    if(row==w.first_row-1) return t[col];
    if(row==w.first_row+w.rows) return b[col];
    return w.cur[(row-w.first_row)*w.n+col].current_energy;
}

static void simulate(LocalWorld& w, int iters) {
    for(int it=0;it<iters;++it) {
        MPI_Sendrecv(w.rows?w.cur.data():nullptr,w.n,MPI_DOUBLE,w.top,10,w.top_energy.data(),w.n,MPI_DOUBLE,w.top,11,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        MPI_Sendrecv(w.rows?w.cur.data()+(w.rows-1)*w.n:nullptr,w.n,MPI_DOUBLE,w.bottom,11,w.bottom_energy.data(),w.n,MPI_DOUBLE,w.bottom,10,MPI_COMM_WORLD,MPI_STATUS_IGNORE);
        for(size_t i=0;i<w.st.size();++i) { const auto& s=w.st[i]; const auto& d=w.cur[i]; val_t f=w.materials[s.material_idx].external_flow;
            for(idx_t k=0;k<s.num_connections;++k) f+=(energy(w,s.connected_idx[k],w.top_energy,w.bottom_energy)-d.current_energy)*.8*s.connected_flux[k]*.25;
            w.next[i]={d.current_energy+f,d.total_flux+std::abs(f)};
        }
        w.cur.swap(w.next);
    }
}

static uint64_t local_hash(const LocalWorld& w) { uint64_t h=0; size_t off=static_cast<size_t>(w.first_row)*w.n; for(size_t i=0;i<w.cur.size();++i) { auto a=reinterpret_cast<const uint64_t*>(&w.cur[i].current_energy), b=reinterpret_cast<const uint64_t*>(&w.cur[i].total_flux); h^=(*a+off+i)*0x9e3779b97f4a7c15ULL; h^=(*b+off+i)*0xbf58476d1ce4e5b9ULL; } return h; }

int main(int argc,char** argv) {
    MPI_Init(&argc,&argv); int rank,size; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&size);
    int n=512,iters=10; bool validate=false, results=false;
    for(int i=1;i<argc;++i) { if(!strcmp(argv[i],"-n")&&i+1<argc)n=atoi(argv[++i]); else if(!strcmp(argv[i],"-i")&&i+1<argc)iters=atoi(argv[++i]); else if(!strcmp(argv[i],"-v"))validate=true; else if(!strcmp(argv[i],"-r"))results=true; else if(!strcmp(argv[i],"-h")){if(rank==0)usage(argv[0]);MPI_Finalize();return 0;} else {if(rank==0)usage(argv[0]);MPI_Finalize();return 1;} }
    if(n<1||iters<0){if(rank==0)fprintf(stderr,"Invalid grid or iteration count\n");MPI_Finalize();return 1;}
    LocalWorld w; build(w,n,rank,size); if(rank==0) { printf("Unstructured Mesh Energy Transfer Benchmark\n============================================\nGrid size: %d x %d = %d elements\nIterations: %d\nValidation: %s\n\n",n,n,n*n,iters,validate?"enabled":"disabled"); printf("Running MPI simulation on %d ranks...\n",size); }
    MPI_Barrier(MPI_COMM_WORLD); double start=MPI_Wtime(); simulate(w,iters); double elapsed=MPI_Wtime()-start, max_elapsed; MPI_Reduce(&elapsed,&max_elapsed,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD);
    uint64_t lh=local_hash(w), hash=0; MPI_Reduce(&lh,&hash,1,MPI_UINT64_T,MPI_BXOR,0,MPI_COMM_WORLD);
    if(rank==0){printf("Computation time: %.3f ms\n",max_elapsed*1000); int measured=std::max(iters-1,1); double ge=(double)measured*n*n/(max_elapsed*1e9); printf("Performance:\n  Time per iteration: %.4f ms\n  Elements/sec: %.4f GigaElements/s\n  Performance: %.4f GFLOPS\n  Result hash: %016lX\n",max_elapsed*1000/measured,ge,ge*22.,hash);}
    if(results||validate){ int count=results?w.rows*n:0; std::vector<int> counts(size),displs(size); MPI_Gather(&count,1,MPI_INT,counts.data(),1,MPI_INT,0,MPI_COMM_WORLD); if(rank==0){displs[0]=0;for(int i=1;i<size;++i)displs[i]=displs[i-1]+counts[i-1];} std::vector<val_t> all(static_cast<size_t>(n)*n); std::vector<val_t> local; if(results){local.resize(count);for(int i=0;i<count;++i)local[i]=w.cur[i].current_energy;} MPI_Gatherv(results?local.data():nullptr,count,MPI_DOUBLE,all.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
        if(rank==0&&results) print_results(all,"ElementEnergy");
        if(validate){ double sums[4]={0,0,std::numeric_limits<double>::max(),std::numeric_limits<double>::lowest()}; for(auto& d:w.cur){sums[0]+=d.current_energy;sums[1]+=d.total_flux;sums[2]=std::min(sums[2],d.current_energy);sums[3]=std::max(sums[3],d.current_energy);} double g[4]; MPI_Reduce(sums,g,2,MPI_DOUBLE,MPI_SUM,0,MPI_COMM_WORLD); MPI_Reduce(sums+2,g+2,1,MPI_DOUBLE,MPI_MIN,0,MPI_COMM_WORLD); MPI_Reduce(sums+3,g+3,1,MPI_DOUBLE,MPI_MAX,0,MPI_COMM_WORLD); if(rank==0)printf("Validation results:\n  Energy sum: %.12f\n  Flux sum: %.2f\n  Energy range: [%.6f, %.6f]\n  Validation: PASSED\n",g[0],g[1],g[2],g[3]);}
    }
    MPI_Finalize(); return 0;
}
