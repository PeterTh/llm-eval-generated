// QT Clustering Benchmark - Hybrid MPI + OpenMP + CUDA Version
//
// QT (Quality Threshold) clustering is an algorithm that builds clusters
// by starting with a seed point and iteratively adding the closest point
// that maintains the cluster's diameter below a threshold.
//
// Parallelization strategy
// ------------------------
//  * MPI    : the candidate seeds are distributed round-robin across ranks.
//             Each rank evaluates the candidate clusters of its seeds on its
//             GPU; the globally best cluster is selected with an MPI_MAXLOC
//             reduction (largest cardinality, lowest seed index on ties, which
//             is exactly the selection order of the sequential scan) and its
//             member list is broadcast by the owning rank.
//  * CUDA   : the candidate cluster of every seed is grown by one thread block
//             (dynamic work queue, largest candidate sets first). Each block
//             keeps, for every still-admissible candidate, its running maximum
//             distance to the cluster members, so one growth step costs O(C)
//             instead of O(C * members); candidates that can never be added
//             (max distance already >= threshold) are dropped. Member lists
//             are recorded in a device pool.
//  * OpenMP : host bookkeeping (selection, cache maintenance, result
//             serialisation, validation) and the host fallback that regrows a
//             cluster when its member list could not be kept on the device.
//
// Exactness
// ---------
//  * The result of a seed's greedy growth depends only on the set of
//    unclustered points within `threshold` of the seed, and removing a point
//    that was never chosen cannot alter any argmin. A cached result is thus
//    reused until one of its own members (or, if the member list is not
//    available, any point within the threshold) gets clustered.
//  * A seed whose candidate count + 1 is below the best cardinality already
//    known in the current iteration can never be selected and is skipped.
//  * Distances use the identical IEEE double operation sequence on host and
//    device (explicit fma, contraction disabled): d = sqrt(fma(dx,dx,dy*dy)).
//    The GPU works on the squared value q (sqrt is monotone, so comparisons
//    are equivalent; "d < threshold" becomes "q <= qThreshold"). Because
//    rounding of sqrt could merge two different q into the same d and change
//    the lowest-index tie-break, a potential tie is detected and the seed is
//    redone with rounded sqrt distances, exactly as the reference computes.
//  * The GPU growth is guided by single-precision values with a rigorous
//    error bound (coordinates relative to the seed): every comparison whose
//    float gap does not exceed the bound is resolved exactly in double, so
//    the FP64 work (and the FP64 pipe, 1/64 rate on consumer GPUs) is only
//    needed for the genuinely ambiguous cases.

#include <algorithm>
#include <chrono>
#include <cfloat>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include "../common/results_output.hpp"

static const double MAX_WIDTH = 20.0;
static const double MAX_HEIGHT = 20.0;

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err__ = (call);                                               \
        if (err__ != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n", cudaGetErrorString(err__), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

// Structure to represent a point in 2D space
struct Point {
    double x, y;
};

// Structure to represent a cluster
struct Cluster {
    std::vector<int> members;
    int seed_point;
};

// Generate synthetic 2D point data in clusters
void generateSyntheticData(std::vector<Point>& points, const int N, unsigned int seed = 42) {
    auto frand = [&seed]() mutable { return rand_r(&seed) / static_cast<double>(RAND_MAX); };
    
    const double min_dim = std::min(MAX_WIDTH, MAX_HEIGHT);
    int count = 0;
    
    while (count < N) {
        // Create group_cnt points within a circle of radius R
        // around center point (cntr_x, cntr_y)
        const double cntr_x = frand() * MAX_WIDTH;
        const double cntr_y = frand() * MAX_HEIGHT;
        const double R = frand() * min_dim / 2.0;
        int group_cnt = static_cast<int>(frand() * (N / 30.0));
        
        // Make sure we don't make more points than we need
        if (group_cnt > (N - count)) {
            group_cnt = N - count;
        }
        
        while (group_cnt > 0) {
            const double sign = (frand() < 0.5) ? -1.0 : 1.0;
            const double r = frand() * R;
            const double dx = (2.0 * frand() - 1.0) * r;
            // r*r - dx*dx evaluated as a fused multiply-subtract (same as the
            // reference build) so the generated data set is bit-identical.
            const double dy = std::sqrt(std::fma(r, r, -(dx * dx))) * sign;
            const double x = cntr_x + dx;
            const double y = cntr_y + dy;
            
            if (x < 0 || x > MAX_WIDTH || y < 0 || y > MAX_HEIGHT) {
                continue;
            }
            
            points[count] = {x, y};
            count++;
            group_cnt--;
        }
    }
}

// Squared distance before the final sqrt: q = fma(dx, dx, dy*dy). The
// reference distance is exactly sqrt(q), correctly rounded, on host and device.
__host__ __device__ inline double qdist2d(const double x1, const double y1,
                                          const double x2, const double y2) {
    const double dx = x1 - x2;
    const double dy = y1 - y2;
    return fma(dx, dx, dy * dy);
}

// Euclidean distance; identical operation sequence on host and device.
__host__ __device__ inline double dist2d(const double x1, const double y1,
                                         const double x2, const double y2) {
    return sqrt(qdist2d(x1, y1, x2, y2));
}

inline double distance(const Point& p1, const Point& p2) {
    return dist2d(p1.x, p1.y, p2.x, p2.y);
}

// Largest q such that sqrt(q) < threshold. Since RN(sqrt(.)) is monotone,
// sqrt(q) < threshold  <=>  q <= qThreshold(threshold), exactly.
static double qThreshold(const double threshold) {
    double q = threshold * threshold;
    while (std::sqrt(q) < threshold) q = std::nextafter(q, HUGE_VAL);
    while (std::sqrt(q) >= threshold) q = std::nextafter(q, -HUGE_VAL);
    return q;
}

// Two candidates with squared max distances q1 <= q2 can only have the same
// rounded sqrt if q2 <= q1 * (1 + 2^-45): sqrt halves the relative gap and a
// gap of more than 2 ulp survives rounding to nearest. Used (conservatively)
// to detect potential ties that need the exact sqrt-based evaluation.
static const double TIE_FACTOR = 1.0 + 1.0 / 35184372088832.0;  // 1 + 2^-45

// ---------------------------------------------------------------------------
// CUDA kernels
// ---------------------------------------------------------------------------

static const int BLOCK_THREADS = 256;
static const int SHM_CAP = 1024;   // candidates kept in shared memory per block
static const int INV_CHUNK = 1024; // members staged in shared memory per pass

__device__ inline void warpArgmin(double& v, int& i) {
    for (int off = 16; off > 0; off >>= 1) {
        const double ov = __shfl_down_sync(0xffffffffu, v, off);
        const int oi = __shfl_down_sync(0xffffffffu, i, off);
        if (ov < v || (ov == v && oi < i)) {
            v = ov;
            i = oi;
        }
    }
}

// Block-wide argmin over (value, index) pairs, lexicographic order.
// Returns the index (INT_MAX if nothing was found) to every thread and
// leaves the winning value in v.
__device__ inline int blockArgmin(double& v, int i, double* s_v, int* s_i) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    warpArgmin(v, i);
    if (lane == 0) {
        s_v[wid] = v;
        s_i[wid] = i;
    }
    __syncthreads();
    if (wid == 0) {
        v = (lane < nwarps) ? s_v[lane] : DBL_MAX;
        i = (lane < nwarps) ? s_i[lane] : INT_MAX;
        warpArgmin(v, i);
        if (lane == 0) {
            s_i[0] = i;
            s_v[0] = v;
        }
    }
    __syncthreads();
    const int result = s_i[0];
    v = s_v[0];
    __syncthreads();
    return result;
}

// Count, for every seed job, the admissible candidates (unclustered points
// other than the seed within the threshold). One block per seed.
__global__ void __launch_bounds__(BLOCK_THREADS)
countCandidatesKernel(const double2* __restrict__ pts,
                      const float2* __restrict__ ptsf,
                      const unsigned char* __restrict__ clustered,
                      const int N,
                      const double qthr,
                      const float qthrf,
                      const float ferr,
                      const int* __restrict__ seeds,
                      const int num_seeds,
                      int* __restrict__ cnt_out) {
    __shared__ int s_count;
    const int job = blockIdx.x;
    if (job >= num_seeds) return;
    if (threadIdx.x == 0) s_count = 0;
    __syncthreads();
    const int seed = seeds[job];
    const double2 sp = pts[seed];
    const float2 sf = ptsf[seed];
    int local = 0;
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        if (i == seed || clustered[i]) continue;
        const float2 pf = ptsf[i];
        const float fdx = pf.x - sf.x;
        const float fdy = pf.y - sf.y;
        if (fdx * fdx + fdy * fdy - ferr > qthrf) continue;  // certainly outside
        const double2 p = pts[i];
        if (qdist2d(p.x, p.y, sp.x, sp.y) <= qthr) local++;
    }
    for (int off = 16; off > 0; off >>= 1) local += __shfl_down_sync(0xffffffffu, local, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(&s_count, local);
    __syncthreads();
    if (threadIdx.x == 0) cnt_out[job] = s_count;
}

// Block-wide minimum of a float, result to every thread.
__device__ inline float blockMinFloat(float v, float* s_f) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    for (int off = 16; off > 0; off >>= 1) v = fminf(v, __shfl_down_sync(0xffffffffu, v, off));
    if (lane == 0) s_f[wid] = v;
    __syncthreads();
    if (wid == 0) {
        v = (lane < nwarps) ? s_f[lane] : INFINITY;
        for (int off = 16; off > 0; off >>= 1) v = fminf(v, __shfl_down_sync(0xffffffffu, v, off));
        if (lane == 0) s_f[0] = v;
    }
    __syncthreads();
    const float r = s_f[0];
    __syncthreads();
    return r;
}

// Block-wide argmin over (key, index) pairs in lexicographic order, where key
// is the bit pattern of a non-negative double (order preserving), so the
// reduction runs entirely on the integer pipes. Returns the index (INT_MAX if
// nothing was found) to every thread and leaves the winning key in k.
__device__ inline void warpArgminKey(unsigned long long& k, int& i) {
    for (int off = 16; off > 0; off >>= 1) {
        const unsigned long long ok = __shfl_down_sync(0xffffffffu, k, off);
        const int oi = __shfl_down_sync(0xffffffffu, i, off);
        if (ok < k || (ok == k && oi < i)) {
            k = ok;
            i = oi;
        }
    }
}

__device__ inline int blockArgminKey(unsigned long long& k, int i,
                                     unsigned long long* s_k, int* s_i) {
    const int lane = threadIdx.x & 31;
    const int wid = threadIdx.x >> 5;
    const int nwarps = blockDim.x >> 5;
    warpArgminKey(k, i);
    if (lane == 0) {
        s_k[wid] = k;
        s_i[wid] = i;
    }
    __syncthreads();
    if (wid == 0) {
        k = (lane < nwarps) ? s_k[lane] : ULLONG_MAX;
        i = (lane < nwarps) ? s_i[lane] : INT_MAX;
        warpArgminKey(k, i);
        if (lane == 0) {
            s_i[0] = i;
            s_k[0] = k;
        }
    }
    __syncthreads();
    const int result = s_i[0];
    k = s_k[0];
    __syncthreads();
    return result;
}

static const int LIST_CAP = 256;  // deferred-work lists per block

// Per-candidate state of the fast (single-precision guided) growth.
struct FastStore {
    int* idx;     // point index (-1: dead slot)
    float2* xf;   // float coordinates relative to the seed
    float* mf;    // float approximation of the exact max squared distance
    int* ms;      // member realising that maximum
};

// Shared-memory scratch of the fast growth.
struct FastShared {
    int* plist;                 // slots that may hold the minimum
    int* dlist;                 // slots whose update needs an exact decision
    int2* tlist;                // (point, old member) potential ties
    int* counters;              // [0] plist [1] dlist [2] tlist [3] tie flag [4] materialisation
    unsigned int* minbits;      // float bits of the minimum over deferred slots
    unsigned long long* rk;     // reduction scratch (32)
    int* ri;                    // reduction scratch (32)
    float* rf;                  // reduction scratch (32)
};

// Fast growth of one seed's candidate cluster by the whole block.
//
// Exact state of a candidate c is md(c) = max over members m of q(c, m)
// (double, q = fma(dx,dx,dy*dy)). It is represented by the member ms(c)
// realising the maximum, so md(c) = q(c, ms(c)) can be recomputed exactly on
// demand, plus a float approximation mf(c) with |mf - md| <= ferr_rel. All
// decisions (new maximum, threshold, argmin, tie detection) are taken in
// single precision when the float gap exceeds the error bound; the rare
// ambiguous cases are collected in lists and resolved exactly in double with
// all lanes busy (FP64 instructions are issued per warp, so resolving them
// inline would cost a full warp slot each). The result is bit-identical to
// the exact evaluation. Returns the cardinality; sets tie=true (block-uniform)
// if two candidates' rounded sqrt values might have tied, which requires the
// exact sqrt-based evaluation.
__device__ __forceinline__ int growFast(const FastStore st,
                                        const FastShared sh,
                                        const int C,
                                        const int seed,
                                        const double2* __restrict__ pts,
                                        const float2* __restrict__ ptsf,
                                        const unsigned char* __restrict__ clustered,
                                        const int N,
                                        const double qthr,
                                        const float qthrf,
                                        const float ferr,
                                        const float ferr_rel,
                                        int* __restrict__ mem,
                                        bool& tie_out) {
    const int tid = threadIdx.x;
    const int B = blockDim.x;
    const double2 sp = pts[seed];
    const float2 sf = ptsf[seed];
    const float win = 3.0f * ferr_rel;  // near-minimum window
    const unsigned int INF_BITS = __float_as_uint(INFINITY);

    if (tid == 0) {
        mem[0] = seed;
        sh.counters[0] = 0;
        sh.counters[1] = 0;
        sh.counters[2] = 0;
        sh.counters[3] = 0;
        sh.counters[4] = 0;
        *sh.minbits = INF_BITS;
    }
    __syncthreads();
    // Materialise the candidate list; the only member is the seed.
    for (int i = tid; i < N; i += B) {
        if (i == seed || clustered[i]) continue;
        const float2 pf = ptsf[i];
        const float fdx = pf.x - sf.x;
        const float fdy = pf.y - sf.y;
        if (fdx * fdx + fdy * fdy - ferr > qthrf) continue;  // certainly outside
        const double2 p = pts[i];
        const double q = qdist2d(p.x, p.y, sp.x, sp.y);
        if (q <= qthr) {
            const int k = atomicAdd(&sh.counters[4], 1);
            st.idx[k] = i;
            // Coordinates relative to the seed: |.| < threshold, so the float
            // error of the squared distances is bounded by ferr_rel.
            st.xf[k] = make_float2(static_cast<float>(p.x - sp.x), static_cast<float>(p.y - sp.y));
            st.mf[k] = static_cast<float>(q);
            st.ms[k] = seed;
        }
    }
    __syncthreads();

    // Thread t owns slots t, t+B, t+2B, ... and keeps its still admissible
    // candidates compacted in its first n_t slots.
    int n_t = (C > tid) ? (C - tid + B - 1) / B : 0;
    float lminf = INFINITY;
    for (int j = 0; j < n_t; ++j) lminf = fminf(lminf, st.mf[tid + j * B]);
    float minf = blockMinFloat(lminf, sh.rf);
    bool had_def = false;

    int card = 1;
    bool tie_local = false;
    for (;;) {
        // 1. Collect the candidates that may hold the exact minimum.
        const float lim = minf + win;
        unsigned long long bk = ULLONG_MAX;
        int bi = INT_MAX;
        if (tid == 0) {
            sh.counters[1] = 0;
            sh.counters[2] = 0;
            *sh.minbits = INF_BITS;
        }
        if (lminf <= lim || had_def)
        for (int j = 0; j < n_t; ++j) {
            const int k = tid + j * B;
            if (st.mf[k] <= lim) {
                const int pos = atomicAdd(&sh.counters[0], 1);
                if (pos < LIST_CAP) {
                    sh.plist[pos] = k;
                } else {  // list full: resolve inline
                    const int i = st.idx[k];
                    const double2 p = pts[i];
                    const double2 m = pts[st.ms[k]];
                    const unsigned long long key =
                        static_cast<unsigned long long>(__double_as_longlong(qdist2d(p.x, p.y, m.x, m.y)));
                    if (key < bk || (key == bk && i < bi)) {
                        bk = key;
                        bi = i;
                    }
                }
            }
        }
        __syncthreads();
        // 2. Exact argmin over the collected candidates.
        const int pc = min(sh.counters[0], LIST_CAP);
        for (int e = tid; e < pc; e += B) {
            const int k = sh.plist[e];
            const int i = st.idx[k];
            const double2 p = pts[i];
            const double2 m = pts[st.ms[k]];
            const unsigned long long key =
                static_cast<unsigned long long>(__double_as_longlong(qdist2d(p.x, p.y, m.x, m.y)));
            if (key < bk || (key == bk && i < bi)) {
                bk = key;
                bi = i;
            }
        }
        unsigned long long ck = bk;
        const int chosen = blockArgminKey(ck, bi, sh.rk, sh.ri);
        if (chosen == INT_MAX) break;
        const double chosen_v = __longlong_as_double(static_cast<long long>(ck));
        if (tid == 0) {
            mem[card] = chosen;
            sh.counters[0] = 0;
        }
        card++;

        // 3. Sweep: fold the new member into every candidate, drop those
        //    whose maximum certainly reached the threshold, defer ambiguous
        //    cases, compact, track the float minimum.
        const double2 npd = pts[chosen];
        const float2 nf = make_float2(static_cast<float>(npd.x - sp.x), static_cast<float>(npd.y - sp.y));
        const float tie_limitf = minf + win;  // bound for md <= chosen_v * TIE_FACTOR
        lminf = INFINITY;
        had_def = false;
        int w = 0;
        int nx_i = 0, nx_ms = 0;
        float nx_mf = 0.0f;
        float2 nx_pf = make_float2(0.0f, 0.0f);
        if (n_t > 0) {
            nx_i = st.idx[tid];
            nx_mf = st.mf[tid];
            nx_ms = st.ms[tid];
            nx_pf = st.xf[tid];
        }
        for (int j = 0; j < n_t; ++j) {
            const int k = tid + j * B;
            const int i = nx_i;
            float mf = nx_mf;
            int ms = nx_ms;
            const float2 pf = nx_pf;
            if (j + 1 < n_t) {  // prefetch the next slot (own slots: no hazard)
                const int kn = k + B;
                nx_i = st.idx[kn];
                nx_mf = st.mf[kn];
                nx_ms = st.ms[kn];
                nx_pf = st.xf[kn];
            }
            if (i < 0 || i == chosen) continue;  // dead slot / now a member
            // Tie detection for the choice just made: a lower-index candidate
            // whose rounded sqrt might equal the chosen one's.
            if (i < chosen && mf <= tie_limitf) {
                const int pos = atomicAdd(&sh.counters[2], 1);
                if (pos < LIST_CAP) {
                    sh.tlist[pos] = make_int2(i, ms);
                } else {
                    const double2 p = pts[i];
                    const double2 m = pts[ms];
                    if (qdist2d(p.x, p.y, m.x, m.y) <= chosen_v * TIE_FACTOR) tie_local = true;
                }
            }
            const float fdx = pf.x - nf.x;
            const float fdy = pf.y - nf.y;
            const float qf = fdx * fdx + fdy * fdy;
            bool defer = false;
            if (qf > mf + 2.0f * ferr_rel) {
                mf = qf;            // certainly the new maximum
                ms = chosen;
            } else if (qf >= mf - 2.0f * ferr_rel) {
                defer = true;       // ambiguous maximum
            }
            bool keep = true;
            if (!defer) {
                if (mf + ferr_rel <= qthrf - ferr_rel) {
                    keep = true;
                } else if (mf - ferr_rel > qthrf + ferr_rel) {
                    keep = false;
                } else {
                    defer = true;   // ambiguous threshold
                }
            }
            if (keep) {
                const int kw = tid + w * B;
                if (kw != k) {
                    st.idx[kw] = i;
                    st.xf[kw] = pf;
                }
                st.mf[kw] = mf;
                st.ms[kw] = ms;
                ++w;
                if (!defer) {
                    lminf = fminf(lminf, mf);
                } else {
                    const int pos = atomicAdd(&sh.counters[1], 1);
                    if (pos < LIST_CAP) {
                        sh.dlist[pos] = kw;
                        had_def = true;
                    } else {  // list full: resolve inline
                        const double2 p = pts[i];
                        const double2 np = pts[chosen];
                        const double2 m = pts[ms];
                        const double qn = qdist2d(p.x, p.y, np.x, np.y);
                        const double qo = qdist2d(p.x, p.y, m.x, m.y);
                        double md = qo;
                        if (qn > qo) {
                            md = qn;
                            st.ms[kw] = chosen;
                        }
                        if (md <= qthr) {
                            const float mdf = static_cast<float>(md);
                            st.mf[kw] = mdf;
                            lminf = fminf(lminf, mdf);
                        } else {
                            st.idx[kw] = -1;
                            st.mf[kw] = INFINITY;
                        }
                    }
                }
            }
            // else: max distance reached the threshold, dropped for good
        }
        n_t = w;
        __syncthreads();

        // 4. Resolve the deferred work exactly, all lanes busy.
        const int tc = min(sh.counters[2], LIST_CAP);
        for (int e = tid; e < tc; e += B) {
            const int2 t = sh.tlist[e];
            const double2 p = pts[t.x];
            const double2 m = pts[t.y];
            if (qdist2d(p.x, p.y, m.x, m.y) <= chosen_v * TIE_FACTOR) sh.counters[3] = 1;
        }
        const int dc = min(sh.counters[1], LIST_CAP);
        if (dc > 0) {
            const double2 np = pts[chosen];
            for (int e = tid; e < dc; e += B) {
                const int kw = sh.dlist[e];
                const int i = st.idx[kw];
                const double2 p = pts[i];
                const double2 m = pts[st.ms[kw]];
                const double qn = qdist2d(p.x, p.y, np.x, np.y);
                const double qo = qdist2d(p.x, p.y, m.x, m.y);
                double md = qo;
                if (qn > qo) {
                    md = qn;
                    st.ms[kw] = chosen;
                }
                if (md <= qthr) {
                    const float mdf = static_cast<float>(md);
                    st.mf[kw] = mdf;
                    atomicMin(sh.minbits, __float_as_uint(mdf));
                } else {
                    st.idx[kw] = -1;
                    st.mf[kw] = INFINITY;
                }
            }
        }
        __syncthreads();
        const float dmin = __uint_as_float(*sh.minbits);
        minf = fminf(blockMinFloat(lminf, sh.rf), dmin);
    }
    tie_out = (__syncthreads_or((tie_local || sh.counters[3] != 0) ? 1 : 0) != 0);
    return card;
}

// Exact (reference-equivalent) growth with rounded sqrt distances; used only
// when the fast path detected a potential tie. Candidate coordinates are
// fetched from pts on demand.
__device__ __forceinline__ int growExact(int* __restrict__ cidx,
                                         double* __restrict__ cmd,
                                         const int C,
                                         const int seed,
                                         const double2* __restrict__ pts,
                                         const unsigned char* __restrict__ clustered,
                                         const int N,
                                         const double threshold,
                                         const double qthr,
                                         int* __restrict__ mem,
                                         int* s_count,
                                         double* s_rv,
                                         int* s_ri) {
    const int tid = threadIdx.x;
    const int B = blockDim.x;
    const double2 sp = pts[seed];
    if (tid == 0) {
        *s_count = 0;
        mem[0] = seed;
    }
    __syncthreads();
    for (int i = tid; i < N; i += B) {
        if (i == seed || clustered[i]) continue;
        const double2 p = pts[i];
        const double q = qdist2d(p.x, p.y, sp.x, sp.y);
        if (q <= qthr) {
            const int k = atomicAdd(s_count, 1);
            cidx[k] = i;
            cmd[k] = sqrt(q);
        }
    }
    __syncthreads();
    int n_t = (C > tid) ? (C - tid + B - 1) / B : 0;
    double bv = DBL_MAX;
    int bi = INT_MAX;
    for (int j = 0; j < n_t; ++j) {
        const int k = tid + j * B;
        const double md = cmd[k];
        const int i = cidx[k];
        if (md < bv || (md == bv && i < bi)) {
            bv = md;
            bi = i;
        }
    }
    int card = 1;
    for (;;) {
        double chosen_v = bv;
        const int chosen = blockArgmin(chosen_v, bi, s_rv, s_ri);
        if (chosen == INT_MAX) break;
        if (tid == 0) mem[card] = chosen;
        card++;
        const double2 np = pts[chosen];
        bv = DBL_MAX;
        bi = INT_MAX;
        int w = 0;
        for (int j = 0; j < n_t; ++j) {
            const int k = tid + j * B;
            const int i = cidx[k];
            if (i == chosen) continue;
            double md = cmd[k];
            const double2 p = pts[i];
            const double d = sqrt(qdist2d(p.x, p.y, np.x, np.y));
            if (d > md) md = d;
            if (md < threshold) {
                const int kw = tid + w * B;
                cidx[kw] = i;
                cmd[kw] = md;
                ++w;
                if (md < bv || (md == bv && i < bi)) {
                    bv = md;
                    bi = i;
                }
            }
        }
        n_t = w;
    }
    return card;
}

// One thread block grows the candidate cluster of one seed at a time (seeds
// are taken from a dynamic work queue, ordered by decreasing candidate count).
// Produces the cardinality and records the member list in `pool` (offset
// written to off_out, -1 if the pool is full). `bound` holds the largest
// cardinality found so far (a lower bound of this iteration's maximum): a seed
// whose upper bound cannot reach it is skipped and reported as -1.
__global__ void __launch_bounds__(BLOCK_THREADS, 4)
candidateClusterKernel(const double2* __restrict__ pts,
                       const float2* __restrict__ ptsf,
                       const unsigned char* __restrict__ clustered,
                       const int N,
                       const double threshold,
                       const double qthr,
                       const float ferr,
                       const float ferr_rel,
                       const int* __restrict__ seeds,
                       const int* __restrict__ cnts,
                       const int* __restrict__ ubs,
                       const int num_seeds,
                       int* __restrict__ card_out,
                       long long* __restrict__ off_out,
                       int* __restrict__ g_idx,
                       float2* __restrict__ g_xf,
                       float* __restrict__ g_mf,
                       int* __restrict__ g_ms,
                       double* __restrict__ g_md,
                       int* __restrict__ g_mem,
                       const size_t g_stride,
                       int* __restrict__ pool,
                       const long long pool_cap,
                       unsigned long long* __restrict__ pool_used,
                       int* __restrict__ work_counter,
                       int* __restrict__ bound) {
    extern __shared__ unsigned char smem_raw[];
    float2* s_xf = reinterpret_cast<float2*>(smem_raw);
    float* s_mf = reinterpret_cast<float*>(s_xf + SHM_CAP);
    int* s_ms = reinterpret_cast<int*>(s_mf + SHM_CAP);
    int* s_idx = reinterpret_cast<int*>(s_ms + SHM_CAP);
    __shared__ int s_plist[LIST_CAP];
    __shared__ int s_dlist[LIST_CAP];
    __shared__ int2 s_tlist[LIST_CAP];
    __shared__ int s_counters[8];
    __shared__ unsigned int s_minbits;
    __shared__ unsigned long long s_rk[32];
    __shared__ double s_rv[32];
    __shared__ int s_ri[32];
    __shared__ float s_rf[32];
    __shared__ int s_job;
    __shared__ int s_bound;
    __shared__ long long s_off;
    const FastShared sh = {s_plist, s_dlist, s_tlist, s_counters, &s_minbits, s_rk, s_ri, s_rf};

    const int tid = threadIdx.x;
    const int B = blockDim.x;
    const size_t boff = static_cast<size_t>(blockIdx.x) * g_stride;
    int* mem = g_mem + boff;
    const float qthrf = static_cast<float>(qthr);

    for (;;) {
        if (tid == 0) {
            s_job = atomicAdd(work_counter, 1);
            s_bound = *reinterpret_cast<volatile int*>(bound);
        }
        __syncthreads();
        const int job = s_job;
        if (job >= num_seeds) break;

        const int seed = seeds[job];
        const int C = cnts[job];

        // Known upper bound cannot beat the current best: skip.
        // (s_bound is read by one thread so the decision is block-uniform.)
        if (ubs[job] < s_bound) {
            if (tid == 0) {
                card_out[job] = -1;
                off_out[job] = -1;
            }
            __syncthreads();
            continue;
        }

        int card;
        bool tie;
        if (C <= SHM_CAP) {
            FastStore st = {s_idx, s_xf, s_mf, s_ms};
            card = growFast(st, sh, C, seed, pts, ptsf, clustered, N, qthr, qthrf, ferr, ferr_rel, mem, tie);
        } else {
            FastStore st = {g_idx + boff, g_xf + boff, g_mf + boff, g_ms + boff};
            card = growFast(st, sh, C, seed, pts, ptsf, clustered, N, qthr, qthrf, ferr, ferr_rel, mem, tie);
        }
        if (tie) {
            // Potential rounding tie: redo with rounded sqrt distances.
            card = growExact(g_idx + boff, g_md + boff, C, seed, pts, clustered, N,
                             threshold, qthr, mem, &s_counters[4], s_rv, s_ri);
        }

        // Publish: cardinality, raise the bound, store the member list.
        if (tid == 0) {
            card_out[job] = card;
            atomicMax(bound, card);
            const unsigned long long off =
                atomicAdd(pool_used, static_cast<unsigned long long>(card));
            s_off = (static_cast<long long>(off) + card <= pool_cap) ? static_cast<long long>(off) : -1;
            off_out[job] = s_off;
        }
        __syncthreads();
        const long long off = s_off;
        if (off >= 0) {
            for (int m = tid; m < card; m += B) pool[off + m] = mem[m];
        }
        __syncthreads();
    }
}

// Candidate-count invalidation: a seed's candidate set changed if any newly
// clustered member lies within the threshold of the seed.
__global__ void invalidateCountKernel(const double2* __restrict__ pts,
                                      const float2* __restrict__ ptsf,
                                      const unsigned char* __restrict__ clustered,
                                      const int N,
                                      const double qthr,
                                      const float ferr,
                                      const int* __restrict__ members,
                                      const int M,
                                      unsigned char* __restrict__ valid) {
    __shared__ float2 s_mf[INV_CHUNK];
    __shared__ int s_mi[INV_CHUNK];
    const float qthrf = static_cast<float>(qthr);
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const bool alive = (i < N) && !clustered[i] && valid[i];
    double2 p = make_double2(0.0, 0.0);
    float2 pf = make_float2(0.0f, 0.0f);
    if (alive) {
        p = pts[i];
        pf = ptsf[i];
    }
    bool inv = false;
    for (int base = 0; base < M; base += INV_CHUNK) {
        const int n = min(INV_CHUNK, M - base);
        __syncthreads();
        for (int j = threadIdx.x; j < n; j += blockDim.x) {
            const int m = members[base + j];
            s_mi[j] = m;
            s_mf[j] = ptsf[m];
        }
        __syncthreads();
        if (alive && !inv) {
            for (int j = 0; j < n; ++j) {
                const float2 mf = s_mf[j];
                const float fdx = pf.x - mf.x;
                const float fdy = pf.y - mf.y;
                if (fdx * fdx + fdy * fdy - ferr > qthrf) continue;  // certainly outside
                const double2 m = pts[s_mi[j]];
                if (qdist2d(p.x, p.y, m.x, m.y) <= qthr) {
                    inv = true;
                    break;
                }
            }
        }
    }
    if (alive && inv) valid[i] = 0;
}

// Cardinality invalidation for seeds with a recorded member list: the cached
// result is stale iff one of its members has been clustered. One warp per seed.
__global__ void invalidateMembersKernel(const int* __restrict__ seeds,
                                        const int num_seeds,
                                        const long long* __restrict__ moff,
                                        const int* __restrict__ mcard,
                                        const int* __restrict__ pool,
                                        const unsigned char* __restrict__ clustered,
                                        unsigned char* __restrict__ card_valid) {
    const int warp = (blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= num_seeds) return;
    const int s = seeds[warp];
    const long long off = moff[s];
    const int card = mcard[s];
    bool hit = false;
    for (int m = lane; m < card; m += 32) {
        if (clustered[pool[off + m]]) {
            hit = true;
            break;
        }
    }
    if (__any_sync(0xffffffffu, hit) && lane == 0) card_valid[s] = 0;
}

// ---------------------------------------------------------------------------
// Host side
// ---------------------------------------------------------------------------

// Host fallback: regrow the member list of a seed whose list was not kept on
// the device. Same greedy procedure (exact sqrt distances), OpenMP-parallel
// over the candidates.
static void reconstructCluster(const int seed,
                               const std::vector<Point>& points,
                               const std::vector<unsigned char>& clustered,
                               const double threshold,
                               std::vector<int>& members) {
    const int N = static_cast<int>(points.size());
    const Point sp = points[seed];

    // Build the candidate list (order irrelevant, ties are broken on index).
    std::vector<int> cidx;
    std::vector<double> cmd;
    std::vector<Point> cxy;
    #pragma omp parallel
    {
        std::vector<int> li;
        std::vector<double> ld;
        std::vector<Point> lp;
        #pragma omp for schedule(static) nowait
        for (int i = 0; i < N; ++i) {
            if (i == seed || clustered[i]) continue;
            const double d = distance(points[i], sp);
            if (d < threshold) {
                li.push_back(i);
                ld.push_back(d);
                lp.push_back(points[i]);
            }
        }
        #pragma omp critical
        {
            cidx.insert(cidx.end(), li.begin(), li.end());
            cmd.insert(cmd.end(), ld.begin(), ld.end());
            cxy.insert(cxy.end(), lp.begin(), lp.end());
        }
    }
    const int C = static_cast<int>(cidx.size());

    members.clear();
    members.push_back(seed);

    // Few candidates: barriers would dominate, use fewer threads.
    const int nthreads = std::max(1, std::min(omp_get_max_threads(), C / 512));
    std::vector<double> tv(static_cast<size_t>(nthreads) * 8, DBL_MAX);
    std::vector<int> ti(static_cast<size_t>(nthreads) * 8, INT_MAX);
    int g_chosen = INT_MAX;

    #pragma omp parallel num_threads(nthreads)
    {
        const int t = omp_get_thread_num();
        for (;;) {
            double bv = DBL_MAX;
            int bi = INT_MAX;
            #pragma omp for schedule(static) nowait
            for (int k = 0; k < C; ++k) {
                const double md = cmd[k];
                if (md < threshold) {
                    const int i = cidx[k];
                    if (md < bv || (md == bv && i < bi)) {
                        bv = md;
                        bi = i;
                    }
                }
            }
            tv[static_cast<size_t>(t) * 8] = bv;
            ti[static_cast<size_t>(t) * 8] = bi;
            #pragma omp barrier
            #pragma omp single
            {
                double gv = DBL_MAX;
                int gi = INT_MAX;
                const int nt = omp_get_num_threads();
                for (int q = 0; q < nt; ++q) {
                    const double v = tv[static_cast<size_t>(q) * 8];
                    const int i = ti[static_cast<size_t>(q) * 8];
                    if (v < gv || (v == gv && i < gi)) {
                        gv = v;
                        gi = i;
                    }
                }
                g_chosen = gi;
                if (gi != INT_MAX) members.push_back(gi);
            }
            const int chosen = g_chosen;
            if (chosen == INT_MAX) break;
            const Point np = points[chosen];
            #pragma omp for schedule(static)
            for (int k = 0; k < C; ++k) {
                const double md = cmd[k];
                if (md < threshold) {
                    if (cidx[k] == chosen) {
                        cmd[k] = DBL_MAX;
                    } else {
                        const double d = distance(cxy[k], np);
                        if (d > md) cmd[k] = d;
                    }
                }
            }
        }
    }
}

struct GpuContext {
    double2* d_pts = nullptr;
    float2* d_ptsf = nullptr;
    unsigned char* d_clustered = nullptr;
    unsigned char* d_valid = nullptr;
    unsigned char* d_card_valid = nullptr;
    int* d_seeds = nullptr;
    int* d_cnts = nullptr;
    int* d_ubs = nullptr;
    int* d_card = nullptr;
    long long* d_off = nullptr;
    int* d_members = nullptr;
    long long* d_moff = nullptr;   // per seed: member list offset in pool
    int* d_mcard = nullptr;        // per seed: cardinality
    int* d_counter = nullptr;
    int* d_bound = nullptr;
    unsigned long long* d_pool_used = nullptr;
    int* d_pool = nullptr;
    long long pool_cap = 0;
    int* d_sidx = nullptr;
    float2* d_sxf = nullptr;
    float* d_smf = nullptr;
    int* d_sms = nullptr;
    double* d_smd = nullptr;
    int* d_smem = nullptr;
    int max_blocks = 0;
    size_t shm_bytes = 0;
};

static void initGpu(GpuContext& g, const std::vector<Point>& points) {
    const int N = static_cast<int>(points.size());
    g.shm_bytes = static_cast<size_t>(SHM_CAP) * (sizeof(float2) + sizeof(float) + 2 * sizeof(int));

    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    int blocks_per_sm = 0;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
        &blocks_per_sm, candidateClusterKernel, BLOCK_THREADS, g.shm_bytes));
    if (blocks_per_sm < 1) blocks_per_sm = 1;
    g.max_blocks = prop.multiProcessorCount * blocks_per_sm;

    // Per-block scratch (spill area for candidate lists larger than SHM_CAP
    // and the member list): keep it within a quarter of the free memory.
    size_t free_b = 0, total_b = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    const size_t per_block = static_cast<size_t>(N) * (sizeof(float2) + sizeof(float) + sizeof(double) + 3 * sizeof(int));
    const int max_by_mem = static_cast<int>(std::max<size_t>(1, (free_b / 4) / std::max<size_t>(per_block, 1)));
    if (g.max_blocks > max_by_mem) g.max_blocks = max_by_mem;

    CUDA_CHECK(cudaMalloc(&g.d_pts, sizeof(double2) * N));
    CUDA_CHECK(cudaMalloc(&g.d_ptsf, sizeof(float2) * N));
    CUDA_CHECK(cudaMalloc(&g.d_clustered, N));
    CUDA_CHECK(cudaMalloc(&g.d_valid, N));
    CUDA_CHECK(cudaMalloc(&g.d_card_valid, N));
    CUDA_CHECK(cudaMalloc(&g.d_seeds, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&g.d_cnts, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&g.d_ubs, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&g.d_card, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&g.d_off, sizeof(long long) * N));
    CUDA_CHECK(cudaMalloc(&g.d_members, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&g.d_moff, sizeof(long long) * N));
    CUDA_CHECK(cudaMalloc(&g.d_mcard, sizeof(int) * N));
    CUDA_CHECK(cudaMalloc(&g.d_counter, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d_bound, sizeof(int)));
    CUDA_CHECK(cudaMalloc(&g.d_pool_used, sizeof(unsigned long long)));
    const size_t stride = static_cast<size_t>(N);
    CUDA_CHECK(cudaMalloc(&g.d_sidx, sizeof(int) * stride * g.max_blocks));
    CUDA_CHECK(cudaMalloc(&g.d_sxf, sizeof(float2) * stride * g.max_blocks));
    CUDA_CHECK(cudaMalloc(&g.d_smf, sizeof(float) * stride * g.max_blocks));
    CUDA_CHECK(cudaMalloc(&g.d_sms, sizeof(int) * stride * g.max_blocks));
    CUDA_CHECK(cudaMalloc(&g.d_smd, sizeof(double) * stride * g.max_blocks));
    CUDA_CHECK(cudaMalloc(&g.d_smem, sizeof(int) * stride * g.max_blocks));

    // Member-list pool: a quarter of what is left, at most 2 GiB.
    CUDA_CHECK(cudaMemGetInfo(&free_b, &total_b));
    size_t pool_bytes = std::min<size_t>(free_b / 4, static_cast<size_t>(2) << 30);
    pool_bytes = std::max<size_t>(pool_bytes, static_cast<size_t>(16) << 20);
    g.pool_cap = static_cast<long long>(pool_bytes / sizeof(int));
    CUDA_CHECK(cudaMalloc(&g.d_pool, sizeof(int) * static_cast<size_t>(g.pool_cap)));

    static_assert(sizeof(Point) == sizeof(double2), "Point must match double2 layout");
    CUDA_CHECK(cudaMemcpy(g.d_pts, points.data(), sizeof(double2) * N, cudaMemcpyHostToDevice));
    {
        std::vector<float> pf(static_cast<size_t>(N) * 2);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            pf[2 * static_cast<size_t>(i)] = static_cast<float>(points[i].x);
            pf[2 * static_cast<size_t>(i) + 1] = static_cast<float>(points[i].y);
        }
        CUDA_CHECK(cudaMemcpy(g.d_ptsf, pf.data(), sizeof(float2) * N, cudaMemcpyHostToDevice));
    }
    CUDA_CHECK(cudaMemset(g.d_clustered, 0, N));
    CUDA_CHECK(cudaMemset(g.d_valid, 0, N));
    CUDA_CHECK(cudaMemset(g.d_card_valid, 0, N));
    CUDA_CHECK(cudaMemset(g.d_pool_used, 0, sizeof(unsigned long long)));
}

static void freeGpu(GpuContext& g) {
    cudaFree(g.d_pts);
    cudaFree(g.d_ptsf);
    cudaFree(g.d_clustered);
    cudaFree(g.d_valid);
    cudaFree(g.d_card_valid);
    cudaFree(g.d_seeds);
    cudaFree(g.d_cnts);
    cudaFree(g.d_ubs);
    cudaFree(g.d_card);
    cudaFree(g.d_off);
    cudaFree(g.d_members);
    cudaFree(g.d_moff);
    cudaFree(g.d_mcard);
    cudaFree(g.d_counter);
    cudaFree(g.d_bound);
    cudaFree(g.d_pool_used);
    cudaFree(g.d_pool);
    cudaFree(g.d_sidx);
    cudaFree(g.d_sxf);
    cudaFree(g.d_smf);
    cudaFree(g.d_sms);
    cudaFree(g.d_smd);
    cudaFree(g.d_smem);
}

// Main QT clustering algorithm (hybrid MPI + OpenMP + CUDA)
std::vector<Cluster> qtClustering(const std::vector<Point>& points,
                                  const double threshold,
                                  const int rank,
                                  const int size) {
    const int N = static_cast<int>(points.size());
    const double qthr = qThreshold(threshold);
    // Error bound of the single-precision pre-filters: with |coordinates| <= M
    // the float squared distance differs from the exact value by less than
    // 48 * M^2 * 2^-24 (conversions, subtractions, squares, sum) and a float
    // copy of a running maximum (<= 8 M^2) by at most 8 * M^2 * 2^-24; use a
    // >3x safety margin on the sum.
    double maxabs = 1.0;
    #pragma omp parallel for reduction(max : maxabs)
    for (int i = 0; i < N; ++i) {
        maxabs = std::max(maxabs, std::max(std::fabs(points[i].x), std::fabs(points[i].y)));
    }
    const float ferr = static_cast<float>(192.0 * maxabs * maxabs / 16777216.0);
    // Inside a candidate cluster all coordinates relative to the seed are
    // bounded by the threshold (and by the coordinate range), which gives a
    // much tighter bound for the growth phase.
    const double relmax = std::max(1e-30, std::min(threshold, 2.0 * maxabs));
    const float ferr_rel = static_cast<float>(192.0 * relmax * relmax / 16777216.0);
    const float qthrf = static_cast<float>(qthr);
    std::vector<unsigned char> clustered(N, 0);
    std::vector<unsigned char> valid(N, 0);      // cached candidate count is current
    std::vector<unsigned char> card_valid(N, 0); // cached cardinality is current
    std::vector<unsigned char> cv_dev(N, 0);     // device-side copy of card_valid
    std::vector<int> cnt(N, 0);                  // cached candidate count per seed
    std::vector<int> ub(N, 0);                   // cardinality upper bound per seed
    std::vector<int> card(N, 0);                 // cached cardinality per seed
    std::vector<long long> moff(N, -1);          // member list offset (-1: none)
    std::vector<Cluster> clusters;

    // Seeds owned by this rank (round-robin for spatial load balance).
    std::vector<int> my_seeds;
    for (int i = rank; i < N; i += size) my_seeds.push_back(i);

    GpuContext g;
    initGpu(g, points);
    unsigned long long pool_used = 0;

    std::vector<int> jobs, job_cnt, job_ub, job_card, chk;
    std::vector<long long> job_off;
    std::vector<std::pair<int, int>> order;
    std::vector<int> best_members;
    int remaining = N;

    // Local (max cardinality, lowest seed on ties) over seeds with a valid
    // cached cardinality, OpenMP-parallel.
    auto localBest = [&](int& best_c, int& best_s) {
        best_c = -1;
        best_s = INT_MAX;
        const int M = static_cast<int>(my_seeds.size());
        #pragma omp parallel
        {
            int bc = -1, bs = INT_MAX;
            #pragma omp for schedule(static) nowait
            for (int m = 0; m < M; ++m) {
                const int s = my_seeds[m];
                if (!card_valid[s]) continue;
                const int c = card[s];
                if (c > bc || (c == bc && s < bs)) {
                    bc = c;
                    bs = s;
                }
            }
            #pragma omp critical
            {
                if (bc > best_c || (bc == best_c && bs < best_s)) {
                    best_c = bc;
                    best_s = bs;
                }
            }
        }
    };

    while (remaining > 0) {
        // 1. Refresh stale candidate counts of my seeds on the GPU.
        jobs.clear();
        for (int s : my_seeds) {
            if (!valid[s]) jobs.push_back(s);
        }
        if (!jobs.empty()) {
            const int J = static_cast<int>(jobs.size());
            CUDA_CHECK(cudaMemcpyAsync(g.d_seeds, jobs.data(), sizeof(int) * J, cudaMemcpyHostToDevice));
            countCandidatesKernel<<<J, BLOCK_THREADS>>>(g.d_pts, g.d_ptsf, g.d_clustered, N, qthr, qthrf,
                                                        ferr, g.d_seeds, J, g.d_cnts);
            CUDA_CHECK(cudaGetLastError());
            job_cnt.resize(J);
            CUDA_CHECK(cudaMemcpy(job_cnt.data(), g.d_cnts, sizeof(int) * J, cudaMemcpyDeviceToHost));
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < J; ++j) {
                cnt[jobs[j]] = job_cnt[j];
                ub[jobs[j]] = job_cnt[j] + 1;
                valid[jobs[j]] = 1;
            }
        }

        // 2. Global lower bound from cached exact cardinalities.
        int lb_c = -1, lb_s = INT_MAX;
        localBest(lb_c, lb_s);
        {
            int in[2] = {lb_c, lb_s};
            int out[2] = {-1, INT_MAX};
            MPI_Allreduce(in, out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
            lb_c = out[0];
            lb_s = out[1];
        }

        // 3. Grow the candidate clusters of seeds that could still beat the
        //    bound, largest candidate sets first (tightens the bound early).
        order.clear();
        for (int s : my_seeds) {
            if (card_valid[s]) continue;
            const int u = ub[s];
            if (u > lb_c || (u == lb_c && s < lb_s)) order.emplace_back(-u, s);
        }
        std::sort(order.begin(), order.end());
        const int J = static_cast<int>(order.size());
        jobs.resize(J);
        job_cnt.resize(J);
        job_ub.resize(J);
        for (int j = 0; j < J; ++j) {
            jobs[j] = order[j].second;
            job_cnt[j] = cnt[jobs[j]];
            job_ub[j] = -order[j].first;
        }
        if (J > 0) {
            // Recycle the member pool when it is more than half full: lists
            // are dropped (those seeds fall back to count-based invalidation).
            if (pool_used > static_cast<unsigned long long>(g.pool_cap / 2)) {
                pool_used = 0;
                CUDA_CHECK(cudaMemsetAsync(g.d_pool_used, 0, sizeof(unsigned long long)));
                std::fill(moff.begin(), moff.end(), -1LL);
            }
            CUDA_CHECK(cudaMemcpyAsync(g.d_seeds, jobs.data(), sizeof(int) * J, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(g.d_cnts, job_cnt.data(), sizeof(int) * J, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(g.d_ubs, job_ub.data(), sizeof(int) * J, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(g.d_bound, &lb_c, sizeof(int), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemsetAsync(g.d_counter, 0, sizeof(int)));
            const int blocks = std::min(g.max_blocks, J);
            candidateClusterKernel<<<blocks, BLOCK_THREADS, g.shm_bytes>>>(
                g.d_pts, g.d_ptsf, g.d_clustered, N, threshold, qthr, ferr, ferr_rel, g.d_seeds, g.d_cnts, g.d_ubs, J,
                g.d_card, g.d_off, g.d_sidx, g.d_sxf, g.d_smf, g.d_sms, g.d_smd, g.d_smem, static_cast<size_t>(N),
                g.d_pool, g.pool_cap, g.d_pool_used, g.d_counter, g.d_bound);
            CUDA_CHECK(cudaGetLastError());
            job_card.resize(J);
            job_off.resize(J);
            CUDA_CHECK(cudaMemcpyAsync(job_card.data(), g.d_card, sizeof(int) * J, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpyAsync(job_off.data(), g.d_off, sizeof(long long) * J, cudaMemcpyDeviceToHost));
            CUDA_CHECK(cudaMemcpy(&pool_used, g.d_pool_used, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
            #pragma omp parallel for schedule(static)
            for (int j = 0; j < J; ++j) {
                const int s = jobs[j];
                if (job_card[j] >= 0) {
                    card[s] = job_card[j];
                    card_valid[s] = 1;
                    moff[s] = job_off[j];
                }
            }
        }

        // 4. Global best: max cardinality, lowest seed index on ties
        //    (seeds without an exact cardinality are provably smaller).
        int local_best_card = -1, local_best_seed = INT_MAX;
        localBest(local_best_card, local_best_seed);
        int in[2] = {local_best_card, local_best_seed};
        int out[2] = {-1, INT_MAX};
        MPI_Allreduce(in, out, 1, MPI_2INT, MPI_MAXLOC, MPI_COMM_WORLD);
        const int best_card = out[0];
        const int best_seed = out[1];
        if (best_seed < 0 || best_card <= 0 || best_seed == INT_MAX) break;

        // 5. Member list of the winner: fetched from the owner's device pool
        //    and broadcast, or (rarely) regrown on the host by every rank.
        const int owner = best_seed % size;
        best_members.resize(best_card);
        int have_list = 0;
        if (rank == owner && moff[best_seed] >= 0) {
            CUDA_CHECK(cudaMemcpy(best_members.data(), g.d_pool + moff[best_seed],
                                  sizeof(int) * best_card, cudaMemcpyDeviceToHost));
            have_list = 1;
        }
        MPI_Bcast(&have_list, 1, MPI_INT, owner, MPI_COMM_WORLD);
        if (have_list) {
            MPI_Bcast(best_members.data(), best_card, MPI_INT, owner, MPI_COMM_WORLD);
        } else {
            reconstructCluster(best_seed, points, clustered, threshold, best_members);
        }

        Cluster cluster;
        cluster.seed_point = best_seed;
        cluster.members = best_members;
        clusters.push_back(cluster);

        for (int m : best_members) clustered[m] = 1;
        remaining -= static_cast<int>(best_members.size());

        // 6. Update device state and invalidate affected cached results.
        const int K = static_cast<int>(best_members.size());
        chk.clear();
        for (int s : my_seeds) {
            if (card_valid[s] && moff[s] >= 0 && !clustered[s]) chk.push_back(s);
        }
        const int Kc = static_cast<int>(chk.size());
        CUDA_CHECK(cudaMemcpyAsync(g.d_clustered, clustered.data(), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(g.d_valid, valid.data(), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(g.d_card_valid, card_valid.data(), N, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpyAsync(g.d_members, best_members.data(), sizeof(int) * K, cudaMemcpyHostToDevice));
        const int inv_blocks = (N + BLOCK_THREADS - 1) / BLOCK_THREADS;
        invalidateCountKernel<<<inv_blocks, BLOCK_THREADS>>>(g.d_pts, g.d_ptsf, g.d_clustered, N, qthr,
                                                             ferr, g.d_members, K, g.d_valid);
        CUDA_CHECK(cudaGetLastError());
        if (Kc > 0) {
            CUDA_CHECK(cudaMemcpyAsync(g.d_seeds, chk.data(), sizeof(int) * Kc, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(g.d_moff, moff.data(), sizeof(long long) * N, cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpyAsync(g.d_mcard, card.data(), sizeof(int) * N, cudaMemcpyHostToDevice));
            const int warps_per_block = BLOCK_THREADS / 32;
            const int mblocks = (Kc + warps_per_block - 1) / warps_per_block;
            invalidateMembersKernel<<<mblocks, BLOCK_THREADS>>>(g.d_seeds, Kc, g.d_moff, g.d_mcard,
                                                                g.d_pool, g.d_clustered, g.d_card_valid);
            CUDA_CHECK(cudaGetLastError());
        }
        CUDA_CHECK(cudaMemcpyAsync(valid.data(), g.d_valid, N, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(cv_dev.data(), g.d_card_valid, N, cudaMemcpyDeviceToHost));
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < N; ++i) {
            if (clustered[i]) {
                card_valid[i] = 0;
            } else if (moff[i] >= 0) {
                card_valid[i] = cv_dev[i];            // exact: a member got clustered
            } else if (!valid[i]) {
                card_valid[i] = 0;                    // conservative: candidate set changed
            }
        }

        // Remove clustered seeds from my list.
        my_seeds.erase(std::remove_if(my_seeds.begin(), my_seeds.end(),
                                      [&clustered](int idx) { return clustered[idx] != 0; }),
                       my_seeds.end());
    }

    freeGpu(g);
    return clusters;
}

// Validation: check that clusters satisfy the QT clustering properties
bool validateClusters(const std::vector<Cluster>& clusters,
                     const std::vector<Point>& points,
                     const double threshold) {
    bool valid = true;
    
    printf("Validating clusters:\n");
    
    // Check each cluster
    for (size_t c = 0; c < clusters.size(); ++c) {
        const auto& cluster = clusters[c];
        double max_diameter = 0.0;
        const int n = static_cast<int>(cluster.members.size());
        
        // Check diameter (max distance between any two points)
        #pragma omp parallel for schedule(dynamic, 16) reduction(max : max_diameter)
        for (int i = 0; i < n; ++i) {
            for (int j = i + 1; j < n; ++j) {
                const double dist = distance(points[cluster.members[i]], 
                                           points[cluster.members[j]]);
                max_diameter = std::max(max_diameter, dist);
            }
        }
        
        if (c < 10) { // Print first 10 clusters
            printf("  Cluster %zu: size=%zu, seed=%d, diameter=%.4f\n", 
                   c, cluster.members.size(), cluster.seed_point, max_diameter);
        }
        
        // Validate diameter is within threshold
        if (max_diameter > threshold * 1.001) { // Allow small numerical error
            printf("ERROR: Cluster %zu has diameter %.4f > threshold %.4f\n", 
                   c, max_diameter, threshold);
            valid = false;
        }
    }
    
    // Check for duplicate memberships
    std::vector<int> membership(points.size(), -1);
    for (size_t c = 0; c < clusters.size(); ++c) {
        for (size_t i = 0; i < clusters[c].members.size(); ++i) {
            const int member = clusters[c].members[i];
            if (membership[member] >= 0) {
                printf("ERROR: Point %d appears in multiple clusters (%d and %zu)\n",
                       member, membership[member], c);
                valid = false;
            }
            membership[member] = static_cast<int>(c);
        }
    }
    
    // Count clustered points
    int clustered_count = 0;
    #pragma omp parallel for reduction(+ : clustered_count)
    for (size_t i = 0; i < membership.size(); ++i) {
        if (membership[i] >= 0) clustered_count++;
    }
    
    printf("Total points: %zu, Clustered: %d, Unclustered: %zu\n",
           points.size(), clustered_count, points.size() - clustered_count);
    
    return valid;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of points (default: 1000)\n");
    printf("  -t <float>   Distance threshold for clustering (default: 2.0)\n");
    printf("  -v           Enable validation\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank = 0, size = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Bind each rank to a GPU of its node (round-robin over local ranks).
    {
        MPI_Comm local;
        MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
        int local_rank = 0;
        MPI_Comm_rank(local, &local_rank);
        MPI_Comm_free(&local);
        int ndev = 0;
        if (cudaGetDeviceCount(&ndev) != cudaSuccess || ndev <= 0) {
            fprintf(stderr, "Error: no CUDA device available on rank %d\n", rank);
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
        CUDA_CHECK(cudaSetDevice(local_rank % ndev));
    }

    int num_points = 1000;
    double threshold = 2.0;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            num_points = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            threshold = atof(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    
    if (num_points <= 0 || threshold <= 0.0) {
        if (rank == 0) {
            printf("Error: Invalid parameters (num_points=%d, threshold=%.2f)\n", 
                   num_points, threshold);
        }
        MPI_Finalize();
        return 1;
    }
    
    if (rank == 0) {
        printf("QT Clustering Benchmark\n");
        printf("Number of points: %d\n", num_points);
        printf("Distance threshold: %.2f\n", threshold);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d, OpenMP threads/rank: %d\n", size, omp_get_max_threads());
    }
    
    // Generate synthetic data (deterministic, identical on every rank)
    std::vector<Point> points(num_points);
    generateSyntheticData(points, num_points);
    
    // Perform QT clustering
    MPI_Barrier(MPI_COMM_WORLD);
    auto cluster_start = std::chrono::high_resolution_clock::now();
    
    const std::vector<Cluster> clusters = qtClustering(points, threshold, rank, size);
    
    auto cluster_end = std::chrono::high_resolution_clock::now();
    const long local_cluster_time = std::chrono::duration_cast<std::chrono::milliseconds>(
        cluster_end - cluster_start).count();
    long max_cluster_time = 0;
    MPI_Reduce(&local_cluster_time, &max_cluster_time, 1, MPI_LONG, MPI_MAX, 0, MPI_COMM_WORLD);
    const auto cluster_time = std::chrono::milliseconds(max_cluster_time);
    
    // Calculate statistics and performance metrics
    int total_clustered = 0;
    int max_cluster_size = 0;
    
    for (size_t i = 0; i < clusters.size(); ++i) {
        const int size_i = static_cast<int>(clusters[i].members.size());
        total_clustered += size_i;
        max_cluster_size = std::max(max_cluster_size, size_i);
    }
    
    const double avg_cluster_size = clusters.empty() ? 0.0 : 
        static_cast<double>(total_clustered) / clusters.size();
    
    if (rank == 0) {
        printf("Clustering time: %ld ms\n", static_cast<long>(cluster_time.count()));
        printf("Clusters found: %zu\n", clusters.size());
        printf("Points clustered: %d / %d (%.1f%%)\n", 
               total_clustered, num_points, 
               100.0 * total_clustered / num_points);
        printf("Average cluster size: %.2f\n", avg_cluster_size);
        printf("Maximum cluster size: %d\n", max_cluster_size);
        
        // Performance metrics
        const double time_sec = cluster_time.count() / 1000.0;
        const double clusters_per_sec = clusters.size() / time_sec;
        const double points_per_sec = num_points / time_sec;
        printf("Performance: %.1f clusters/s, %.1f points/s\n", 
               clusters_per_sec, points_per_sec);
    }
    
    // Print results for external validation
    if (printResults && rank == 0) {
        // Serialize cluster membership for hashing
        std::vector<int> membership(num_points, -1);
        for (size_t c = 0; c < clusters.size(); ++c) {
            const std::vector<int>& mem = clusters[c].members;
            #pragma omp parallel for schedule(static)
            for (size_t i = 0; i < mem.size(); ++i) {
                membership[mem[i]] = static_cast<int>(c);
            }
        }
        std::vector<double> membershipData(num_points);
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < num_points; ++i) {
            membershipData[i] = static_cast<double>(membership[i]);
        }
        print_results(membershipData, "ClusterMembership");
    }
    
    // Validation
    int ret = 0;
    if (validate) {
        int valid_i = 1;
        if (rank == 0) {
            const bool valid = validateClusters(clusters, points, threshold);
            valid_i = valid ? 1 : 0;
            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
            }
        }
        MPI_Bcast(&valid_i, 1, MPI_INT, 0, MPI_COMM_WORLD);
        ret = valid_i ? 0 : 1;
    }
    
    MPI_Finalize();
    return ret;
}
