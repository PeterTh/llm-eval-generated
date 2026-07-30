/**
 * Room Response Simulation Benchmark - MPI Parallel Version
 * 
 * MPI-parallelized room impulse response simulation using radiosity-based
 * wave propagation. Distributes triangle computations across MPI ranks.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;
constexpr size_t RNG_PER_PAIR = 2 * NUM_RAYS;

struct Vec3 {
    val_t x, y, z;
    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}
    Vec3 operator+(const Vec3& o) const { return {x+o.x, y+o.y, z+o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x-o.x, y-o.y, z-o.z}; }
    Vec3 operator*(val_t s) const { return {x*s, y*s, z*s}; }
    Vec3 operator/(val_t s) const { return {x/s, y/s, z/s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }
    val_t dot(const Vec3& o) const { return x*o.x + y*o.y + z*o.z; }
    Vec3 cross(const Vec3& o) const { return {y*o.z-z*o.y, z*o.x-x*o.z, x*o.y-y*o.x}; }
    val_t squaredNorm() const { return x*x + y*y + z*z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const { val_t n=norm(); return n>EPSILON?*this/n:Vec3(); }
    bool operator==(const Vec3& o) const {
        return std::abs(x-o.x)<EPSILON && std::abs(y-o.y)<EPSILON && std::abs(z-o.z)<EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c, _normal;
    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b-a).cross(c-a).normalized()) {}
    Vec3 center() const { return (a+b+c)/3.0f; }
    Vec3 normal() const { return _normal; }
    val_t area() const { Vec3 ab=b-a,ac=c-a; return 0.5f*ab.cross(ac).norm(); }
    bool operator==(const Triangle& o) const { return a==o.a && b==o.b && c==o.c; }
};

bool triangleBoxOverlap(const Vec3& bc, const Vec3& bhs, const Triangle& tri) {
    Vec3 v0=tri.a-bc, v1=tri.b-bc, v2=tri.c-bc;
    Vec3 e0=v1-v0, e1=v2-v1, e2=v0-v2;
    auto mm3=[](val_t a,val_t b,val_t c){return std::make_pair(std::min({a,b,c}),std::max({a,b,c}));};
    auto [mnx,mxx]=mm3(v0.x,v1.x,v2.x); if(mnx>bhs.x||mxx<-bhs.x) return false;
    auto [mny,mxy]=mm3(v0.y,v1.y,v2.y); if(mny>bhs.y||mxy<-bhs.y) return false;
    auto [mnz,mxz]=mm3(v0.z,v1.z,v2.z); if(mnz>bhs.z||mxz<-bhs.z) return false;
    Vec3 tn=e0.cross(e1);
    val_t d=tn.dot(v0);
    val_t r=bhs.x*std::abs(tn.x)+bhs.y*std::abs(tn.y)+bhs.z*std::abs(tn.z);
    if(std::abs(d)>r) return false;
    auto ta=[&](const Vec3& ax){
        val_t p0=ax.dot(v0),p1=ax.dot(v1),p2=ax.dot(v2);
        val_t rr=bhs.x*std::abs(ax.x)+bhs.y*std::abs(ax.y)+bhs.z*std::abs(ax.z);
        auto [mn,mx]=mm3(p0,p1,p2); return !(mn>rr||mx<-rr);
    };
    Vec3 axes[3]={{1,0,0},{0,1,0},{0,0,1}};
    for(const auto& ax:axes) for(const auto& ed:{e0,e1,e2}){
        Vec3 ca=ax.cross(ed);
        if(ca.squaredNorm()>EPSILON && !ta(ca)) return false;
    }
    return true;
}

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound, halfExtent, center;
    std::unique_ptr<Octree> children[8]{};
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles = nullptr;
    Octree() = default;
    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if(triangles.empty()) return;
        minBound=triangles[0].a; maxBound=triangles[0].a;
        for(const auto& tri:triangles)
            for(const auto* v:{&tri.a,&tri.b,&tri.c}){
                minBound.x=std::min(minBound.x,v->x); minBound.y=std::min(minBound.y,v->y);
                minBound.z=std::min(minBound.z,v->z);
                maxBound.x=std::max(maxBound.x,v->x); maxBound.y=std::max(maxBound.y,v->y);
                maxBound.z=std::max(maxBound.z,v->z);
            }
        std::vector<size_t> ai(triangles.size());
        for(size_t i=0;i<triangles.size();++i) ai[i]=i;
        buildNode(ai,minBound,maxBound);
    }
private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nMin, const Vec3& nMax) {
        minBound=nMin; maxBound=nMax; halfExtent=(maxBound-minBound)*0.5f;
        center=(minBound+maxBound)*0.5f;
        if(indices.size()<=MAX_OCTREE_TRIS||(maxBound-minBound).norm()<MAX_OCTREE_LEAF_SIZE){
            triangleIndices=indices; return;
        }
        Vec3 chs=halfExtent*0.5f;
        std::vector<size_t> ci[8];
        for(size_t idx:indices){
            const Triangle& tri=(*allTriangles)[idx];
            for(int i=0;i<8;++i){
                Vec3 cc=center;
                cc.x+=(i&1)?chs.x:-chs.x; cc.y+=(i&2)?chs.y:-chs.y; cc.z+=(i&4)?chs.z:-chs.z;
                if(triangleBoxOverlap(cc,chs,tri)) ci[i].push_back(idx);
            }
        }
        bool cs=false;
        for(int i=0;i<8;++i) if(!ci[i].empty()&&ci[i].size()<indices.size()){cs=true;break;}
        if(!cs){triangleIndices=indices;return;}
        for(int i=0;i<8;++i){
            if(!ci[i].empty()){
                Vec3 cMin=center,cMax=center;
                cMin.x=(i&1)?center.x:minBound.x; cMax.x=(i&1)?maxBound.x:center.x;
                cMin.y=(i&2)?center.y:minBound.y; cMax.y=(i&2)?maxBound.y:center.y;
                cMin.z=(i&4)?center.z:minBound.z; cMax.z=(i&4)?maxBound.z:center.z;
                children[i]=std::make_unique<Octree>();
                children[i]->allTriangles=allTriangles;
                children[i]->buildNode(ci[i],cMin,cMax);
            }
        }
    }
public:
    bool rayIntersectsBox(const Vec3& p1,const Vec3& p2) const {
        Vec3 d=(p2-p1)*0.5f,c=p1+d-center,ad={std::abs(d.x),std::abs(d.y),std::abs(d.z)};
        if(std::abs(c.x)>halfExtent.x+ad.x) return false;
        if(std::abs(c.y)>halfExtent.y+ad.y) return false;
        if(std::abs(c.z)>halfExtent.z+ad.z) return false;
        if(std::abs(d.y*c.z-d.z*c.y)>halfExtent.y*ad.z+halfExtent.z*ad.y+EPSILON) return false;
        if(std::abs(d.z*c.x-d.x*c.z)>halfExtent.z*ad.x+halfExtent.x*ad.z+EPSILON) return false;
        if(std::abs(d.x*c.y-d.y*c.x)>halfExtent.x*ad.y+halfExtent.y*ad.x+EPSILON) return false;
        return true;
    }
    template<typename Func>
    bool applyToTris(const Vec3& p1,const Vec3& p2,Func&& func) const {
        if(!triangleIndices.empty()){
            for(size_t idx:triangleIndices) if(func(idx,(*allTriangles)[idx])) return true;
            return false;
        }
        for(int i=0;i<8;++i)
            if(children[i]&&children[i]->rayIntersectsBox(p1,p2)&&children[i]->applyToTris(p1,p2,func))
                return true;
        return false;
    }
};

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;
    IcosphereMesh(int subdivisions, val_t radius) {
        const val_t t=(1.0f+std::sqrt(5.0f))/2.0f;
        std::vector<Vec3> verts={
            Vec3(-1,t,0).normalized()*radius,Vec3(1,t,0).normalized()*radius,
            Vec3(-1,-t,0).normalized()*radius,Vec3(1,-t,0).normalized()*radius,
            Vec3(0,-1,t).normalized()*radius,Vec3(0,1,t).normalized()*radius,
            Vec3(0,-1,-t).normalized()*radius,Vec3(0,1,-t).normalized()*radius,
            Vec3(t,0,-1).normalized()*radius,Vec3(t,0,1).normalized()*radius,
            Vec3(-t,0,-1).normalized()*radius,Vec3(-t,0,1).normalized()*radius};
        std::vector<std::array<idx_t,3>> faces={
            {0,11,5},{0,5,1},{0,1,7},{0,7,10},{0,10,11},
            {1,5,9},{5,11,4},{11,10,2},{10,7,6},{7,1,8},
            {3,9,4},{3,4,2},{3,2,6},{3,6,8},{3,8,9},
            {4,9,5},{2,4,11},{6,2,10},{8,6,7},{9,8,1}};
        for(int i=0;i<subdivisions;++i){
            std::vector<std::array<idx_t,3>> nf;
            std::map<std::pair<idx_t,idx_t>,idx_t> mc;
            auto gm=[&](idx_t i1,idx_t i2)->idx_t{
                auto key=std::make_pair(std::min(i1,i2),std::max(i1,i2));
                auto it=mc.find(key); if(it!=mc.end()) return it->second;
                Vec3 mid=(verts[i1]+verts[i2])/2.0f; mid=mid.normalized()*radius;
                idx_t idx=static_cast<idx_t>(verts.size()); verts.push_back(mid);
                mc[key]=idx; return idx;
            };
            for(const auto& f:faces){
                idx_t a=gm(f[0],f[1]),b=gm(f[1],f[2]),c=gm(f[2],f[0]);
                nf.push_back({f[0],a,c}); nf.push_back({f[1],b,a});
                nf.push_back({f[2],c,b}); nf.push_back({a,b,c});
            }
            faces=std::move(nf);
        }
        triangles.reserve(faces.size());
        for(const auto& f:faces)
            triangles.emplace_back(verts[f[2]],verts[f[1]],verts[f[0]]);
    }
};

class RandomGenerator {
    std::mt19937 rng; std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed=42):rng(seed),dist(0.0f,1.0f){}
    val_t rand(){return dist(rng);}
};

val_t rayTriangleIntersect(const Vec3& orig,const Vec3& dir,
    const Vec3& v0,const Vec3& v1,const Vec3& v2){
    Vec3 e1=v1-v0,e2=v2-v0,pvec=dir.cross(e2);
    val_t det=e1.dot(pvec);
    if(std::abs(det)<EPSILON) return std::numeric_limits<val_t>::max();
    val_t invD=1.0f/det; Vec3 tv=orig-v0;
    val_t u=tv.dot(pvec)*invD;
    if(u<0.0f||u>1.0f) return std::numeric_limits<val_t>::max();
    Vec3 qv=tv.cross(e1); val_t v=dir.dot(qv)*invD;
    if(v<0.0f||u+v>1.0f) return std::numeric_limits<val_t>::max();
    return e2.dot(qv)*invD;
}

bool isRayBlocked(const Vec3& from,const Vec3& to,const Octree& octree,
                  size_t sI,size_t dI){
    Vec3 dir=to-from; val_t rl=dir.norm();
    if(rl<EPSILON) return true;
    return octree.applyToTris(from,to,[&](size_t idx,const Triangle& tri){
        if(idx==sI||idx==dI) return false;
        val_t d=rayTriangleIntersect(from,dir/rl,tri.a,tri.b,tri.c);
        return d>EPSILON&&d<rl-EPSILON;
    });
}

val_t cosPhi(const Vec3& v, const Vec3& n){
    val_t vn=v.norm(); if(vn<=EPSILON) return ZERO;
    return std::max(ZERO,v.dot(n)/vn);
}

val_t computeKij(size_t iI,size_t iJ,
    const std::vector<Triangle>& tris, const Octree& octree,
    const val_t* rngBuf, size_t rngOff){
    const Triangle& tI=tris[iI],&tJ=tris[iJ];
    if(tI.normal().dot(tJ.normal())>0.99f) return ZERO;
    val_t kij=ZERO;
    for(int r=0;r<NUM_RAYS;++r){
        val_t u0=rngBuf[rngOff+r*4],v0=rngBuf[rngOff+r*4+1];
        if(u0+v0>1.0f){u0=1.0f-u0;v0=1.0f-v0;}
        Vec3 pI=tI.a+(tI.b-tI.a)*u0+(tI.c-tI.a)*v0;
        val_t u1=rngBuf[rngOff+r*4+2],v1=rngBuf[rngOff+r*4+3];
        if(u1+v1>1.0f){u1=1.0f-u1;v1=1.0f-v1;}
        Vec3 pJ=tJ.a+(tJ.b-tJ.a)*u1+(tJ.c-tJ.a)*v1;
        if(isRayBlocked(pI,pJ,octree,iI,iJ)) continue;
        Vec3 v=pJ-pI; val_t ds=v.squaredNorm();
        if(ds<EPSILON) continue;
        val_t cI=cosPhi(v,tI.normal()),cJ=cosPhi(-v,tJ.normal());
        if(cI<=ZERO||cJ<=ZERO) continue;
        kij+=(cI*cJ)/(PI*ds);
    }
    return kij*INV_NUM_RAYS;
}

int computeTau(const Triangle& tI,const Triangle& tJ){
    return static_cast<int>(std::ceil((tI.center()-tJ.center()).norm()*INV_WAVE_SPEED));
}

inline void blockDist(size_t n,int rank,int nr,int& ls,int& le){
    size_t base=n/static_cast<size_t>(nr),rem=n%static_cast<size_t>(nr);
    ls=static_cast<int>(base*static_cast<size_t>(rank)+std::min(static_cast<size_t>(rank),rem));
    le=static_cast<int>(base*static_cast<size_t>(rank+1)+std::min(static_cast<size_t>(rank+1),rem));
}

struct SimState{
    size_t N, T;
    std::vector<Triangle> triangles;
    std::vector<val_t> areas, rho, kij, radE, radB, distances;
    std::vector<int> tau;
    Octree octree;
    size_t srcIdx;
    size_t idx2d(size_t i,size_t j) const {return i*N+j;}
    size_t idxTN(size_t t,size_t n) const {return t*N+n;}
};

void initState(SimState& s, int subs, size_t ts, size_t si, val_t rh, int rank){
    IcosphereMesh m(subs,10.0f);
    s.triangles=std::move(m.triangles); s.N=s.triangles.size(); s.T=ts; s.srcIdx=si%s.N;
    if(rank==0) printf("Generated icosphere mesh with %zu triangles\n",s.N);
    if(rank==0) printf("Building octree...\n");
    s.octree.build(s.triangles);
    s.areas.resize(s.N); for(size_t i=0;i<s.N;++i) s.areas[i]=s.triangles[i].area();
    s.rho.resize(s.N,rh);
    s.kij.resize(s.N*s.N,ZERO); s.tau.resize(s.N*s.N,0);
    s.radE.resize(ts*s.N,ZERO); s.radB.resize(ts*s.N,ZERO);
    s.distances.resize(s.N,ZERO);
    for(size_t t=0;t<ts/2;++t) s.radE[s.idxTN(t,s.srcIdx)]=1.0f;
}

void computeFormFactors(SimState& s, int rank, int nr){
    if(rank==0) printf("Computing form factors (Kij)...\n");
    int ls,le; blockDist(s.N,rank,nr,ls,le); int lc=le-ls;
    size_t totRNG=s.N*s.N*RNG_PER_PAIR;
    std::vector<val_t> allRNG;
    if(rank==0){
        RandomGenerator rng(42);
        allRNG.resize(totRNG);
        for(size_t k=0;k<totRNG;++k) allRNG[k]=rng.rand();
    }
    size_t myRS=static_cast<size_t>(ls)*s.N*RNG_PER_PAIR;
    size_t myRC=static_cast<size_t>(lc)*s.N*RNG_PER_PAIR;
    std::vector<val_t> myRNG;
    if(myRC>0){
        myRNG.resize(myRC);
        std::vector<int> rc(nr),dp(nr);
        if(rank==0){
            for(int r=0;r<nr;++r){
                int a,b; blockDist(s.N,r,nr,a,b);
                rc[r]=(b-a)*static_cast<int>(s.N*RNG_PER_PAIR);
                dp[r]=a*static_cast<int>(s.N*RNG_PER_PAIR);
            }
        }
        MPI_Scatterv(allRNG.data(),rc.data(),dp.data(),MPI_FLOAT,
                     myRNG.data(),static_cast<int>(myRC),MPI_FLOAT,0,MPI_COMM_WORLD);
    }
    for(int ii=ls;ii<le;++ii){
        size_t i=static_cast<size_t>(ii);
        for(size_t j=0;j<s.N;++j){
            if(i==j) continue;
            size_t ro=(i*s.N+j)*RNG_PER_PAIR-myRS;
            s.kij[s.idx2d(i,j)]=computeKij(i,j,s.triangles,s.octree,myRNG.data(),ro);
        }
        if(rank==0&&((ii+1)%100==0||ii+1==static_cast<int>(s.N)))
            printf("  Progress: %d/%zu triangles\n",ii+1,s.N);
    }
    // Allgatherv Kij
    std::vector<int> rc(nr),dp(nr);
    {int off=0;for(int r=0;r<nr;++r){int a,b;blockDist(s.N,r,nr,a,b);
        rc[r]=(b-a)*static_cast<int>(s.N);dp[r]=off;off+=(b-a)*static_cast<int>(s.N);}}
    int tot=0;for(int r=0;r<nr;++r)tot+=rc[r];
    std::vector<val_t> rb(tot);
    MPI_Allgatherv(s.kij.data()+static_cast<size_t>(ls)*s.N,
        static_cast<int>(static_cast<size_t>(lc)*s.N),MPI_FLOAT,
        rb.data(),rc.data(),dp.data(),MPI_FLOAT,MPI_COMM_WORLD);
    std::vector<val_t> assembled(s.N*s.N,ZERO);
    {int off=0;for(int r=0;r<nr;++r){int a,b;blockDist(s.N,r,nr,a,b);
        for(int ii=a;ii<b;++ii){
            std::memcpy(assembled.data()+static_cast<size_t>(ii)*s.N,
                rb.data()+static_cast<size_t>(off),sizeof(val_t)*s.N);
            off+=static_cast<int>(s.N);}}}
    s.kij=std::move(assembled);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) printf("Form factor computation complete.\n");
}

void computeTimeDelays(SimState& s, int rank, int nr){
    if(rank==0) printf("Computing time delays (Tau)...\n");
    int ls,le; blockDist(s.N,rank,nr,ls,le); int lc=le-ls;
    for(int ii=ls;ii<le;++ii){
        size_t i=static_cast<size_t>(ii);
        for(size_t j=0;j<s.N;++j){
            if(i==j) continue;
            s.tau[s.idx2d(i,j)]=computeTau(s.triangles[i],s.triangles[j]);
        }
    }
    std::vector<int> rc(nr),dp(nr);
    {int off=0;for(int r=0;r<nr;++r){int a,b;blockDist(s.N,r,nr,a,b);
        rc[r]=(b-a)*static_cast<int>(s.N);dp[r]=off;off+=(b-a)*static_cast<int>(s.N);}}
    int tot=0;for(int r=0;r<nr;++r)tot+=rc[r];
    std::vector<int> rb(tot);
    MPI_Allgatherv(s.tau.data()+static_cast<size_t>(ls)*s.N,
        static_cast<int>(static_cast<size_t>(lc)*s.N),MPI_INT,
        rb.data(),rc.data(),dp.data(),MPI_INT,MPI_COMM_WORLD);
    std::vector<int> assembled(s.N*s.N,0);
    {int off=0;for(int r=0;r<nr;++r){int a,b;blockDist(s.N,r,nr,a,b);
        for(int ii=a;ii<b;++ii){
            std::memcpy(assembled.data()+static_cast<size_t>(ii)*s.N,
                rb.data()+off,sizeof(int)*s.N);
            off+=static_cast<int>(s.N);}}}
    s.tau=std::move(assembled);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) printf("Time delay computation complete.\n");
}

void runSimulation(SimState& s, int rank, int nr){
    if(rank==0) printf("Running wave propagation simulation...\n");
    int ls,le; blockDist(s.N,rank,nr,ls,le); int lc=le-ls;
    std::vector<val_t> localRB(lc);
    for(size_t t=0;t<s.T;++t){
        for(int ii=0;ii<lc;++ii){
            size_t i=static_cast<size_t>(ls+ii);
            val_t sumB=ZERO;
            for(size_t j=0;j<s.N;++j){
                if(i==j) continue;
                int td=s.tau[s.idx2d(i,j)];
                if(static_cast<int>(t)<td) continue;
                val_t kj=s.kij[s.idx2d(i,j)];
                if(kj<=ZERO) continue;
                size_t st=t-static_cast<size_t>(td);
                val_t rj=s.radB[s.idxTN(st,j)];
                if(rj<=ZERO) continue;
                sumB+=std::min(kj*s.areas[j],ONE)*rj;
            }
            localRB[ii]=s.rho[i]*sumB+s.radE[s.idxTN(t,i)];
        }
        std::vector<val_t> recvT(s.N);
        MPI_Allgather(localRB.data(),lc,MPI_FLOAT,recvT.data(),lc,MPI_FLOAT,MPI_COMM_WORLD);
        for(size_t j=0;j<s.N;++j) s.radB[s.idxTN(t,j)]=recvT[j];
        if(rank==0&&((t+1)%10==0||t+1==s.T))
            printf("  Timestep %zu/%zu\n",t+1,s.T);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) printf("Simulation complete.\n");
}

void computeDistances(SimState& s, int rank, int nr){
    if(rank==0) printf("Computing distances via cross-correlation...\n");
    int ls,le; blockDist(s.N,rank,nr,ls,le); int lc=le-ls;
    for(int ii=ls;ii<le;++ii){
        size_t i=static_cast<size_t>(ii);
        val_t maxC=ZERO; int bestT=0;
        for(size_t t=0;t<s.T;++t){
            val_t sum=ZERO;
            for(size_t tt=t;tt<s.T;++tt){
                val_t pB=s.radB[s.idxTN(tt,i)];
                val_t pS=s.radB[s.idxTN(tt-t,s.srcIdx)];
                sum+=pS*pB;
            }
            if(sum>maxC){maxC=sum;bestT=static_cast<int>(t);}
        }
        s.distances[i]=WAVE_SPEED*static_cast<val_t>(bestT);
    }
    std::vector<int> rc(nr),dp(nr);
    {int off=0;for(int r=0;r<nr;++r){int a,b;blockDist(s.N,r,nr,a,b);
        rc[r]=b-a;dp[r]=off;off+=b-a;}}
    std::vector<val_t> allD(s.N);
    MPI_Allgatherv(s.distances.data()+static_cast<size_t>(ls),lc,MPI_FLOAT,
        allD.data(),rc.data(),dp.data(),MPI_FLOAT,MPI_COMM_WORLD);
    s.distances=std::move(allD);
    MPI_Barrier(MPI_COMM_WORLD);
    if(rank==0) printf("Distance computation complete.\n");
}

bool validateResults(const SimState& s){
    printf("\nValidation:\n");
    bool allNN=true;
    val_t minD=std::numeric_limits<val_t>::max(),maxD=std::numeric_limits<val_t>::lowest();
    val_t sumD=ZERO; int nzC=0;
    for(size_t i=0;i<s.N;++i){
        val_t d=s.distances[i];
        if(d<0){allNN=false;printf("  ERROR: Negative distance at triangle %zu: %f\n",i,d);}
        if(!std::isfinite(d)){printf("  ERROR: Non-finite distance at triangle %zu: %f\n",i,d);return false;}
        minD=std::min(minD,d); maxD=std::max(maxD,d); sumD+=d;
        if(d>EPSILON) nzC++;
    }
    printf("  Distance range: [%.4f, %.4f]\n",minD,maxD);
    printf("  Average distance: %.4f\n",sumD/static_cast<val_t>(s.N));
    printf("  Non-zero distances: %d/%zu\n",nzC,s.N);
    val_t srcD=s.distances[s.srcIdx];
    if(srcD>WAVE_SPEED*2) printf("  WARNING: Source triangle distance is non-zero: %.4f\n",srcD);
    int recvE=0;
    for(size_t i=0;i<s.N;++i)
        for(size_t t=0;t<s.T;++t)
            if(s.radB[s.idxTN(t,i)]>EPSILON){recvE++;break;}
    printf("  Triangles receiving energy: %d/%zu\n",recvE,s.N);
    if(recvE==0){printf("  ERROR: No triangles received energy\n");return false;}
    int nzK=0;
    for(size_t i=0;i<s.N*s.N;++i) if(s.kij[i]>EPSILON) nzK++;
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",nzK,s.N*s.N,
           100.0f*nzK/static_cast<val_t>(s.N*s.N));
    if(nzK==0){printf("  ERROR: All form factors zero\n");return false;}
    if(!allNN) return false;
    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const SimState& s){
    uint64_t h=0;
    for(size_t i=0;i<s.N;++i){
        const uint32_t* ptr=reinterpret_cast<const uint32_t*>(&s.distances[i]);
        h^=(static_cast<uint64_t>(*ptr)+i)*0x9e3779b97f4a7c15ULL;
    }
    return h;
}

int getSubdivisionsForTriangleCount(int target){
    int subs=0,tris=20;
    while(tris<target&&subs<6){subs++;tris*=4;}
    return subs;
}

void printUsage(const char* progName){
    printf("Usage: %s [options]\n",progName);
    printf("Options:\n");
    printf("  -n <num>     Target number of triangles (default: 320)\n");
    printf("               Actual count: 20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv){
    int targetTriangles=320, timesteps=50, sourceIdx=0;
    val_t reflectivity=0.8f;
    bool validate=false, printResults=false;

    for(int i=1;i<argc;++i){
        if(strcmp(argv[i],"-n")==0&&i+1<argc) targetTriangles=atoi(argv[++i]);
        else if(strcmp(argv[i],"-t")==0&&i+1<argc) timesteps=atoi(argv[++i]);
        else if(strcmp(argv[i],"-s")==0&&i+1<argc) sourceIdx=atoi(argv[++i]);
        else if(strcmp(argv[i],"-r")==0&&i+1<argc) reflectivity=static_cast<val_t>(atof(argv[++i]));
        else if(strcmp(argv[i],"-v")==0) validate=true;
        else if(strcmp(argv[i],"-o")==0) printResults=true;
        else if(strcmp(argv[i],"-h")==0){printUsage(argv[0]);return 0;}
        else{printf("Unknown option: %s\n",argv[i]);printUsage(argv[0]);return 1;}
    }

    int subdivisions=getSubdivisionsForTriangleCount(targetTriangles);

    // Initialize MPI
    MPI_Init(&argc,&argv);
    int rank, numRanks;
    MPI_Comm_rank(MPI_COMM_WORLD,&rank);
    MPI_Comm_size(MPI_COMM_WORLD,&numRanks);

    if(rank==0){
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("MPI ranks: %d\n",numRanks);
        printf("Target triangles: %d (using %d subdivisions)\n",targetTriangles,subdivisions);
        printf("Timesteps: %d\n",timesteps);
        printf("Source triangle: %d\n",sourceIdx);
        printf("Reflectivity: %.2f\n",reflectivity);
        printf("Validation: %s\n",validate?"enabled":"disabled");
        printf("\n");
    }

    SimState state;
    initState(state,subdivisions,static_cast<size_t>(timesteps),
              static_cast<size_t>(sourceIdx),reflectivity,rank);

    if(rank==0) printf("\n");

    auto startPre=std::chrono::high_resolution_clock::now();
    computeTimeDelays(state,rank,numRanks);
    computeFormFactors(state,rank,numRanks);
    auto endPre=std::chrono::high_resolution_clock::now();
    long preDuration=std::chrono::duration_cast<std::chrono::milliseconds>(endPre-startPre).count();
    if(rank==0) printf("Precomputation time: %ld ms\n\n",preDuration);

    auto startSim=std::chrono::high_resolution_clock::now();
    runSimulation(state,rank,numRanks);
    auto endSim=std::chrono::high_resolution_clock::now();
    long simDuration=std::chrono::duration_cast<std::chrono::milliseconds>(endSim-startSim).count();
    if(rank==0) printf("Simulation time: %ld ms\n\n",simDuration);

    auto startDist=std::chrono::high_resolution_clock::now();
    computeDistances(state,rank,numRanks);
    auto endDist=std::chrono::high_resolution_clock::now();
    long distDuration=std::chrono::duration_cast<std::chrono::milliseconds>(endDist-startDist).count();
    if(rank==0) printf("Distance computation time: %ld ms\n\n",distDuration);

    long totalTime=preDuration+simDuration+distDuration;
    if(rank==0){
        size_t n=state.N,t=state.T;
        printf("Total computation time: %ld ms\n",totalTime);
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n",n);
        printf("  Timesteps: %zu\n",t);
        printf("  Form factor computations: %.2e\n",static_cast<double>(n*n));
        printf("  Simulation operations: %.2e\n",static_cast<double>(n*n*t));
        printf("  Distance computations: %.2e\n",static_cast<double>(n*t*t));
        printf("  Total time per triangle: %.4f ms\n",static_cast<double>(totalTime)/n);
        size_t memKij=n*n*sizeof(val_t),memTau=n*n*sizeof(int);
        size_t memRad=2*t*n*sizeof(val_t),totalMem=memKij+memTau+memRad;
        printf("  Memory usage: %.2f MB\n",totalMem/(1024.0*1024.0));
        uint64_t hash=computeHash(state);
        printf("  Result hash: %016lX\n\n",hash);
    }

    if(printResults && rank==0){
        std::vector<double> distData(state.distances.begin(),state.distances.end());
        print_results(distData,"Distances");
    }

    if(validate && rank==0){
        if(!validateResults(state)) return 1;
    }

    MPI_Finalize();
    return 0;
}
