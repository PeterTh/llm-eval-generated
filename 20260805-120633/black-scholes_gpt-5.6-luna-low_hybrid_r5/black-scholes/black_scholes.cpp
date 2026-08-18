#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include "../common/results_output.hpp"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

enum OptionType { CALL = 0, PUT = 1 };
struct OptionInput {
    int type;
    double strike, spot, q, r, t, vol, value, tol;
};

__host__ __device__ inline double cumulativeNormal(double x) noexcept {
    return 0.5 * (1.0 + erf(x * M_SQRT1_2));
}
__host__ __device__ inline double blackScholes(const OptionInput& o) noexcept {
    if (o.t <= 0.0 || o.vol <= 0.0) return 0.0;
    const double rootT = sqrt(o.t);
    const double d1 = (log(o.spot / o.strike) +
        (o.r - o.q + 0.5 * o.vol * o.vol) * o.t) / (o.vol * rootT);
    const double d2 = d1 - o.vol * rootT;
    const double sDisc = o.spot * exp(-o.q * o.t);
    const double kDisc = o.strike * exp(-o.r * o.t);
    return o.type == CALL ? sDisc * cumulativeNormal(d1) - kDisc * cumulativeNormal(d2)
        : kDisc * cumulativeNormal(-d2) - sDisc * cumulativeNormal(-d1);
}

__global__ void priceKernel(const OptionInput* options, double* results, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) results[i] = blackScholes(options[i]);
}

inline constexpr std::array<OptionInput, 7> testOptions() noexcept {
    return {{
    {CALL,40,42,.04,.08,.75,.35,5.0975,1e-3}, {CALL,100,90,.10,.10,.10,.15,.0205,1e-3},
    {CALL,100,100,.10,.10,.10,.15,1.8734,1e-3}, {CALL,100,110,.10,.10,.10,.15,9.9413,1e-3},
    {PUT,100,90,.10,.10,.10,.15,9.9210,1e-3}, {PUT,100,100,.10,.10,.10,.15,1.8734,1e-3},
    {PUT,100,110,.10,.10,.10,.15,.0408,1e-3}}};
}

void generateOptions(std::vector<OptionInput>& a, size_t first, size_t n) {
    constexpr auto base = testOptions(); a.resize(n);
    #pragma omp parallel for schedule(static)
    for (long long j = 0; j < static_cast<long long>(n); ++j) {
        const size_t i = first + static_cast<size_t>(j);
        a[static_cast<size_t>(j)] = base[i % base.size()];
        const double f = 1.0 + 0.1 * (i / static_cast<double>(base.size()));
        a[static_cast<size_t>(j)].spot *= f; a[static_cast<size_t>(j)].strike *= f;
    }
}

bool validateResults(const std::vector<OptionInput>& o, const std::vector<double>& r) {
    bool ok = true; const int n = static_cast<int>(std::min<size_t>(10, o.size()));
    printf("Checking computed option prices:\n");
    for (int i=0; i<n; ++i) {
        const double e=fabs(r[i]-o[i].value), re=e/(fabs(o[i].value)+1e-10);
        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n",i,r[i],o[i].value,re);
        if (r[i]<0 || r[i]>1000 || std::isnan(r[i]) || std::isinf(r[i])) ok=false;
    }
    return ok;
}
void usage(const char* p) { printf("Usage: %s [-n num] [-v] [-r] [-h]\n",p); }
inline void cudaCheck(cudaError_t e, const char* where) { if(e!=cudaSuccess){fprintf(stderr,"CUDA error at %s: %s\n",where,cudaGetErrorString(e)); MPI_Abort(MPI_COMM_WORLD,1);} }

int main(int argc, char** argv) {
    MPI_Init(&argc,&argv); int rank=0, nr=1; MPI_Comm_rank(MPI_COMM_WORLD,&rank); MPI_Comm_size(MPI_COMM_WORLD,&nr);
    size_t total=10000; bool validate=false, printResults=false;
    for(int i=1;i<argc;++i) { if(!strcmp(argv[i],"-n")&&i+1<argc) total=strtoull(argv[++i],nullptr,10);
        else if(!strcmp(argv[i],"-v")) validate=true; else if(!strcmp(argv[i],"-r")) printResults=true;
        else if(!strcmp(argv[i],"-h")){if(rank==0)usage(argv[0]); MPI_Finalize(); return 0;} else {if(rank==0)usage(argv[0]); MPI_Finalize(); return 1;} }
    const size_t first = total * static_cast<size_t>(rank) / nr, last = total * static_cast<size_t>(rank+1) / nr, localN=last-first;
    std::vector<OptionInput> local; generateOptions(local, first, localN); std::vector<double> localR(localN);
    OptionInput *dO=nullptr; double *dR=nullptr; cudaCheck(cudaSetDevice(rank % std::max(1, [](){int n=0; cudaGetDeviceCount(&n); return n;}())),"device");
    cudaCheck(cudaMalloc(&dO,localN*sizeof(OptionInput)),"malloc options"); cudaCheck(cudaMalloc(&dR,localN*sizeof(double)),"malloc results");
    MPI_Barrier(MPI_COMM_WORLD); const double start=MPI_Wtime();
    cudaCheck(cudaMemcpy(dO,local.data(),localN*sizeof(OptionInput),cudaMemcpyHostToDevice),"copy options");
    priceKernel<<<static_cast<unsigned>((localN+255)/256),256>>>(dO,dR,localN); cudaCheck(cudaGetLastError(),"kernel"); cudaCheck(cudaDeviceSynchronize(),"sync");
    cudaCheck(cudaMemcpy(localR.data(),dR,localN*sizeof(double),cudaMemcpyDeviceToHost),"copy results"); const double elapsed=MPI_Wtime()-start;
    cudaFree(dO); cudaFree(dR);
    std::vector<int> counts(nr), displs(nr); for(int p=0;p<nr;++p){size_t b=total*p/nr,e=total*(p+1)/nr;counts[p]=static_cast<int>(e-b);displs[p]=static_cast<int>(b);}
    std::vector<double> results; std::vector<OptionInput> all; if(rank==0){results.resize(total); all.resize(total);}
    MPI_Gatherv(localR.data(),static_cast<int>(localN),MPI_DOUBLE,results.data(),counts.data(),displs.data(),MPI_DOUBLE,0,MPI_COMM_WORLD);
    std::vector<int> byteCounts(nr), byteDispls(nr); for(int p=0;p<nr;++p){byteCounts[p]=counts[p]*static_cast<int>(sizeof(OptionInput));byteDispls[p]=displs[p]*static_cast<int>(sizeof(OptionInput));}
    MPI_Gatherv(local.data(),static_cast<int>(localN*sizeof(OptionInput)),MPI_BYTE,all.data(),byteCounts.data(),byteDispls.data(),MPI_BYTE,0,MPI_COMM_WORLD);
    if(rank==0){ printf("Black-Scholes Option Pricing Benchmark\nNumber of options: %zu\nValidation: %s\nPricing options...\nComputation time: %.3f ms\nOptions per second: %.0f\n",total,validate?"enabled":"disabled",elapsed*1000,total/elapsed); if(printResults)print_results(results,"OptionPrices"); if(validate) {bool ok=validateResults(all,results); printf("Validation: %s\n",ok?"PASSED":"FAILED"); MPI_Finalize(); return ok?0:1;}}
    MPI_Finalize(); return 0;
}
