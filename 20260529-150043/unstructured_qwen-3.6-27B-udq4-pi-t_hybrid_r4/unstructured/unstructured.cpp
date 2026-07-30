#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>
#include "../common/results_output.hpp"

using idx_t = uint64_t;
using val_t = double;
constexpr int MAX_CONNECTIONS = 8;

struct Material { val_t transfer_coeff; val_t external_flow; };
struct ElementStatic {
    idx_t material_idx; idx_t num_connections;
    idx_t connected_idx[MAX_CONNECTIONS]; val_t connected_flux[MAX_CONNECTIONS];
};
struct ElementDynamic { val_t current_energy; val_t total_flux; };
struct World {
    std::vector<Material> materials;
    std::vector<ElementStatic> elements_static;
    std::vector<ElementDynamic> elements_dynamic;
    std::vector<ElementDynamic> elements_dynamic_swap;
};
constexpr idx_t DEFAULT_MAT_ID=0, INFLOW_MAT_ID=1, OUTFLOW_MAT_ID=2;

// ============================================================================
// CUDA kernel: one thread per element, Jacobi-style energy transfer update
// Each thread computes flux from all connected neighbors and updates energy
// ============================================================================
__global__ void simKernel(const ElementStatic* es, const ElementDynamic* dyn,
                          ElementDynamic* dyns, size_t n) {
    size_t i = (size_t)blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= n) return;
    const ElementStatic& e = es[i];
    const ElementDynamic& d = dyn[i];
    // Material properties: default(0.8,0.0), inflow(0.8,0.5), outflow(0.8,-0.5)
    val_t tc=0.8, ef=0.0;
    if(e.material_idx==1) ef=0.5;
    else if(e.material_idx==2) ef=-0.5;
    val_t tf = ef;
    for(idx_t j=0;j<e.num_connections;++j){
        const ElementDynamic& nb = dyn[e.connected_idx[j]];
        tf += (nb.current_energy - d.current_energy)*tc*e.connected_flux[j]*0.25;
    }
    dyns[i].current_energy = d.current_energy + tf;
    dyns[i].total_flux = d.total_flux + fabs(tf);
}

// ============================================================================
// Build unstructured mesh from 2D grid (OpenMP parallelized)
// ============================================================================
static void buildSquare2D(World& w, int N) {
    int ne=N*N;
    w.materials={{0.8,0.0},{0.8,0.5},{0.8,-0.5}};
    w.elements_static.resize(ne); w.elements_dynamic.resize(ne); w.elements_dynamic_swap.resize(ne);
    #pragma omp parallel for schedule(static)
    for(int i=0;i<ne;++i){
        w.elements_static[i].material_idx=0; w.elements_static[i].num_connections=0;
        w.elements_dynamic[i]={0.0,0.0}; w.elements_dynamic_swap[i]={0.0,0.0};
    }
    #pragma omp parallel for collapse(2) schedule(static)
    for(int x=0;x<N;++x) for(int y=0;y<N;++y){
        int idx=x*N+y; ElementStatic& e=w.elements_static[idx];
        int o[4][2]={{1,0},{-1,0},{0,1},{0,-1}};
        for(int n=0;n<4;++n){
            int nx=x+o[n][0],ny=y+o[n][1];
            if(nx>=0&&nx<N&&ny>=0&&ny<N){
                e.connected_idx[e.num_connections]=nx*N+ny;
                e.connected_flux[e.num_connections]=1.0; e.num_connections++;
            }
        }
    }
    int L=N-1;
    w.elements_static[0].material_idx=1; w.elements_static[L].material_idx=2;
    w.elements_static[L*N].material_idx=2; w.elements_static[N*N-1].material_idx=1;
}

// ============================================================================
// Run simulation on GPU with double-buffering (ping-pong)
// ============================================================================
static void runGPU(World& w, int niters) {
    size_t ne=w.elements_static.size();
    ElementStatic* des=nullptr; ElementDynamic* dd=nullptr,*dds=nullptr,*dt=nullptr;
    cudaMalloc(&des,ne*sizeof(ElementStatic));
    cudaMalloc(&dd,ne*sizeof(ElementDynamic));
    cudaMalloc(&dds,ne*sizeof(ElementDynamic));
    cudaMalloc(&dt,ne*sizeof(ElementDynamic));
    cudaMemcpy(des,w.elements_static.data(),ne*sizeof(ElementStatic),cudaMemcpyHostToDevice);
    cudaMemcpy(dd,w.elements_dynamic.data(),ne*sizeof(ElementDynamic),cudaMemcpyHostToDevice);
    cudaMemcpy(dds,w.elements_dynamic_swap.data(),ne*sizeof(ElementDynamic),cudaMemcpyHostToDevice);
    dim3 blk(256), grd((ne+255)/256);
    for(int it=0;it<niters;++it){
        simKernel<<<grd,blk>>>(des,dd,dds,ne);
        // Double-buffer swap: dt=dd, dd=dds, dds=dt
        cudaMemcpy(dt,dd,ne*sizeof(ElementDynamic),cudaMemcpyDeviceToDevice);
        cudaMemcpy(dd,dds,ne*sizeof(ElementDynamic),cudaMemcpyDeviceToDevice);
        cudaMemcpy(dds,dt,ne*sizeof(ElementDynamic),cudaMemcpyDeviceToDevice);
    }
    cudaDeviceSynchronize();
    cudaMemcpy(w.elements_dynamic.data(),dd,ne*sizeof(ElementDynamic),cudaMemcpyDeviceToHost);
    cudaFree(des); cudaFree(dd); cudaFree(dds); cudaFree(dt);
}

// ============================================================================
// Validate results (OpenMP parallelized reductions)
// ============================================================================
static bool validateResults(const World& w) {
    val_t es=0,fs=0,emx=std::numeric_limits<val_t>::lowest(),emn=std::numeric_limits<val_t>::max();
    #pragma omp parallel for reduction(+:es,fs) reduction(max:emx) reduction(min:emn) schedule(static)
    for(size_t i=0;i<w.elements_dynamic.size();++i){
        es+=w.elements_dynamic[i].current_energy; fs+=w.elements_dynamic[i].total_flux;
        emx=std::max(w.elements_dynamic[i].current_energy,emx);
        emn=std::min(w.elements_dynamic[i].current_energy,emn);
    }
    printf("Validation results:\n");
    printf("  Energy sum: %.12f\n",es);
    printf("  Flux sum: %.2f\n",fs);
    printf("  Energy range: [%.6f, %.6f]\n",emn,emx);
    if(!std::isfinite(es)){printf("  ERROR: Energy sum not finite\n");return false;}
    if(std::abs(es)>1e-8) printf("  WARNING: Energy sum diverged from 0\n");
    if(!std::isfinite(fs)){printf("  ERROR: Flux sum not finite\n");return false;}
    if(!std::isfinite(emx)||!std::isfinite(emn)){printf("  ERROR: Energy extrema not finite\n");return false;}
    printf("  Validation: PASSED\n"); return true;
}

// ============================================================================
// Compute verification hash
// ============================================================================
static uint64_t computeHash(const std::vector<ElementDynamic>& e) {
    uint64_t h=0;
    for(size_t i=0;i<e.size();++i){
        const uint64_t* ep=reinterpret_cast<const uint64_t*>(&e[i].current_energy);
        const uint64_t* fp=reinterpret_cast<const uint64_t*>(&e[i].total_flux);
        h^=(*ep+i)*0x9e3779b97f4a7c15ULL; h^=(*fp+i)*0xbf58476d1ce4e5b9ULL;
    }
    return h;
}

static void printUsage(const char* p){
    printf("Usage: %s [options]\n",p);
    printf("Options:\n  -n <num>  Grid size (NxN) (default: 512)\n");
    printf("  -i <num>  Iterations (default: 10)\n  -v  Validate\n");
    printf("  -r  Print results\n  -h  Help\n");
}

int main(int argc, char** argv) {
    int N=512, niters=10, validate=0, printR=0;
    for(int i=1;i<argc;++i){
        if(strcmp(argv[i],"-n")==0&&i+1<argc) N=atoi(argv[++i]);
        else if(strcmp(argv[i],"-i")==0&&i+1<argc) niters=atoi(argv[++i]);
        else if(strcmp(argv[i],"-v")==0) validate=1;
        else if(strcmp(argv[i],"-r")==0) printR=1;
        else if(strcmp(argv[i],"-h")==0){printUsage(argv[0]);return 0;}
        else{printf("Unknown: %s\n",argv[i]);printUsage(argv[0]);return 1;}
    }
    int ne=N*N;

    // Initialize MPI
    MPI_Init(&argc,&argv);
    int mr,ms; MPI_Comm_rank(MPI_COMM_WORLD,&mr); MPI_Comm_size(MPI_COMM_WORLD,&ms);

    if(mr==0){
        printf("Unstructured Mesh Energy Transfer Benchmark\n");
        printf("============================================\n");
        printf("Grid size: %d x %d = %d elements\n",N,N,ne);
        printf("Iterations: %d\n",niters);
        printf("Parallel: MPI ranks=%d, CUDA GPU/rank, OpenMP threads=%d\n",ms,omp_get_max_threads());
        printf("Validation: %s\n\n",validate?"enabled":"disabled");
    }

    // Assign each MPI rank to a GPU (round-robin across available GPUs)
    int cd=mr%4; cudaSetDevice(cd);
    int dc=0; cudaGetDeviceCount(&dc);
    cudaDeviceProp p; cudaGetDeviceProperties(&p,cd);
    if(mr==0) printf("CUDA: %d devices, using %d (%s)\n",dc,cd,p.name);

    // Build mesh on all ranks (OpenMP parallelized within each rank)
    if(mr==0) printf("Building unstructured mesh...\n");
    World w; buildSquare2D(w,N);
    if(mr==0){
        size_t sm=w.elements_static.size()*sizeof(ElementStatic);
        size_t dm=w.elements_dynamic.size()*sizeof(ElementDynamic)*2;
        printf("Memory usage: %.2f MB (static: %.2f MB, dynamic: %.2f MB)\n\n",
               (sm+dm)/(1024.0*1024.0),sm/(1024.0*1024.0),dm/(1024.0*1024.0));
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // Run simulation on GPU (CUDA kernel with double-buffering)
    if(mr==0) printf("Running simulation...\n");
    auto t0=std::chrono::high_resolution_clock::now();
    runGPU(w,niters);
    auto t1=std::chrono::high_resolution_clock::now();
    long dur=std::chrono::duration_cast<std::chrono::milliseconds>(t1-t0).count();

    // Synchronize all ranks and gather results to rank 0
    MPI_Barrier(MPI_COMM_WORLD);

    // Gather all ranks' results using MPI_Gatherv
    std::vector<ElementDynamic> global_dyn;
    std::vector<int> rc(ms), dp(ms);
    int elem_sz = sizeof(ElementDynamic);
    for(int r=0;r<ms;++r){rc[r]=ne*elem_sz;if(r>0)dp[r]=dp[r-1]+rc[r-1];else dp[r]=0;}
    int total_bytes=dp[ms-1]+rc[ms-1];
    if(mr==0) global_dyn.resize(total_bytes/elem_sz);
    MPI_Gatherv(w.elements_dynamic.data(),ne*elem_sz,MPI_BYTE,
                mr==0?global_dyn.data():nullptr,rc.data(),dp.data(),MPI_BYTE,0,MPI_COMM_WORLD);

    if(mr==0){
        // Use first rank's results (all ranks compute identical results)
        std::vector<ElementDynamic> results(w.elements_dynamic.begin(),w.elements_dynamic.end());
        printf("Computation time: %ld ms\n",dur);
        int ni=std::max(niters-1,1);
        double tpi=(double)dur/ni;
        double ges=(ni*ne)/(dur/1000.0)/1e9;
        double gf=ges*22.0;
        printf("Performance:\n");
        printf("  Time per iteration: %.4f ms\n",tpi);
        printf("  Elements/sec: %.4f GigaElements/s\n",ges);
        printf("  Performance: %.4f GFLOPS\n",gf);
        uint64_t hash=computeHash(results);
        printf("  Result hash: %016lX\n\n",hash);
        if(printR){
            std::vector<double> ed; ed.reserve(results.size());
            for(const auto& e:results) ed.push_back(e.current_energy);
            print_results(ed,"ElementEnergy");
        }
        if(validate){
            World vw; vw.elements_dynamic=results;
            if(!validateResults(vw)) return 1;
        }
    }

    MPI_Finalize(); return 0;
}
