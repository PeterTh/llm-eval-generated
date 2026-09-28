/**
 * Room Response Simulation Benchmark -- CUDA (GPU) parallel implementation
 *
 * This models how sound/light waves propagate between surfaces in a room by
 * computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * Parallelization notes
 * ---------------------
 * All four compute phases run on the GPU:
 *   - Tau      : one thread per (i,j) pair.
 *   - Kij      : NUM_RAYS lanes per (i,j) pair, one shadow ray each, traced
 *                through a flattened (array based) octree.  Keeping a pair's
 *                rays inside one lane group keeps the traversal convergent.
 *   - Radiosity: one thread per receiver triangle, timesteps stay sequential
 *                because radB[t] depends on radB[t-tau].
 *   - Distances: one thread per (triangle, lag) correlation, then an argmax scan.
 *
 * The form factor phase consumes the *exact* std::mt19937 stream of the
 * sequential reference (a single global stream, advanced by 4*NUM_RAYS draws per
 * non-culled pair).  To keep that semantics while running pairs in parallel, the
 * host walks a bit-identical std::mt19937 but only ships one state snapshot per
 * 64 blocks; the device expands those back into the full word stream, tempers
 * them and reproduces libstdc++'s generate_canonical rounding.  Pairs find their
 * slice of the stream through a prefix sum of the "pair consumes randomness"
 * predicate, which is evaluated on the host with the reference's own expression
 * so the two never desynchronize.  Every thread therefore sees precisely the
 * numbers the sequential code would have seen, and all floating point
 * accumulations happen in the original order.
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
#include <thread>
#include <vector>

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t INV_NUM_RAYS = 1.0f / NUM_RAYS;
constexpr val_t EPSILON = 1e-6f;

// Random draws consumed by one non-culled triangle pair (2 per sampled point).
constexpr int DRAWS_PER_PAIR = 4 * NUM_RAYS;

// std::numeric_limits<val_t>::max(), usable from device code
__device__ constexpr float FLT_MAX_DEV = 3.402823466e+38f;

// ============================================================================
// CUDA helpers
// ============================================================================

#define CUDA_CHECK(call)                                                                \
    do {                                                                                \
        cudaError_t _e = (call);                                                        \
        if (_e != cudaSuccess) {                                                        \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(_e),          \
                    __FILE__, __LINE__);                                                \
            exit(1);                                                                    \
        }                                                                               \
    } while (0)

template <typename T>
static T* devAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    return p;
}

// ============================================================================
// Vector and Triangle Types (host side geometry construction)
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    constexpr Vec3() : x(0), y(0), z(0) {}
    constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    val_t squaredNorm() const { return x * x + y * y + z * z; }
    val_t norm() const { return std::sqrt(squaredNorm()); }
    Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction)
// ============================================================================

// Separating axis test for triangle-box overlap
bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    // Translate triangle to box center
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    // Triangle edges
    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    // Test box normals (AABB axes)
    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    // Test triangle normal
    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

    // Test 9 edge cross products
    auto testAxis = [&](const Vec3& axis) {
        val_t p0 = axis.dot(v0);
        val_t p1 = axis.dot(v1);
        val_t p2 = axis.dot(v2);
        val_t r = boxHalfSize.x * std::abs(axis.x) +
                  boxHalfSize.y * std::abs(axis.y) +
                  boxHalfSize.z * std::abs(axis.z);
        auto [minP, maxP] = minMax3(p0, p1, p2);
        return !(minP > r || maxP < -r);
    };

    Vec3 axes[3] = {{1,0,0}, {0,1,0}, {0,0,1}};
    Vec3 edges[3] = {e0, e1, e2};
    for (const auto& axis : axes) {
        for (const auto& edge : edges) {
            Vec3 crossAxis = axis.cross(edge);
            if (crossAxis.squaredNorm() > EPSILON) {
                if (!testAxis(crossAxis)) return false;
            }
        }
    }

    return true;
}

// ============================================================================
// Octree for Spatial Acceleration (built on the host, flattened for the GPU)
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;  // Indices into the global triangle list
    const std::vector<Triangle>* allTriangles;  // Pointer to all triangles

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

        // Compute bounding box
        minBound = triangles[0].a;
        maxBound = triangles[0].a;
        for (const auto& tri : triangles) {
            for (const auto* v : {&tri.a, &tri.b, &tri.c}) {
                minBound.x = std::min(minBound.x, v->x);
                minBound.y = std::min(minBound.y, v->y);
                minBound.z = std::min(minBound.z, v->z);
                maxBound.x = std::max(maxBound.x, v->x);
                maxBound.y = std::max(maxBound.y, v->y);
                maxBound.z = std::max(maxBound.z, v->z);
            }
        }

        // Collect all indices
        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        // If few enough triangles or too small, make this a leaf
        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        // Subdivide into 8 children
        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];

            // Check which children this triangle overlaps
            for (int i = 0; i < 8; ++i) {
                Vec3 childCenter = center;
                childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
                childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
                childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;

                if (triangleBoxOverlap(childCenter, childHalfSize, tri)) {
                    childIndices[i].push_back(idx);
                }
            }
        }

        // Check if we can't split further (all triangles in one child)
        bool canSplit = false;
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty() && childIndices[i].size() < indices.size()) {
                canSplit = true;
                break;
            }
        }

        if (!canSplit) {
            triangleIndices = indices;
            return;
        }

        // Build children
        for (int i = 0; i < 8; ++i) {
            if (!childIndices[i].empty()) {
                Vec3 childMin = center;
                Vec3 childMax = center;
                childMin.x = (i & 1) ? center.x : minBound.x;
                childMax.x = (i & 1) ? maxBound.x : center.x;
                childMin.y = (i & 2) ? center.y : minBound.y;
                childMax.y = (i & 2) ? maxBound.y : center.y;
                childMin.z = (i & 4) ? center.z : minBound.z;
                childMax.z = (i & 4) ? maxBound.z : center.z;

                children[i] = std::make_unique<Octree>();
                children[i]->allTriangles = allTriangles;
                children[i]->buildNode(childIndices[i], childMin, childMax);
            }
        }
    }
};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

// Generate an icosphere by subdividing an icosahedron
class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
        // Initial icosahedron vertices
        const val_t t = (1.0f + std::sqrt(5.0f)) / 2.0f;

        std::vector<Vec3> vertices = {
            Vec3(-1,  t,  0).normalized() * radius,
            Vec3( 1,  t,  0).normalized() * radius,
            Vec3(-1, -t,  0).normalized() * radius,
            Vec3( 1, -t,  0).normalized() * radius,
            Vec3( 0, -1,  t).normalized() * radius,
            Vec3( 0,  1,  t).normalized() * radius,
            Vec3( 0, -1, -t).normalized() * radius,
            Vec3( 0,  1, -t).normalized() * radius,
            Vec3( t,  0, -1).normalized() * radius,
            Vec3( t,  0,  1).normalized() * radius,
            Vec3(-t,  0, -1).normalized() * radius,
            Vec3(-t,  0,  1).normalized() * radius
        };

        // Initial icosahedron faces (20 triangles)
        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

        // Subdivide
        for (int i = 0; i < subdivisions; ++i) {
            std::vector<std::array<idx_t, 3>> newFaces;
            std::map<std::pair<idx_t, idx_t>, idx_t> midpointCache;

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                auto key = std::make_pair(std::min(i1, i2), std::max(i1, i2));
                auto it = midpointCache.find(key);
                if (it != midpointCache.end()) return it->second;

                Vec3 mid = (vertices[i1] + vertices[i2]) / 2.0f;
                mid = mid.normalized() * radius;
                idx_t idx = static_cast<idx_t>(vertices.size());
                vertices.push_back(mid);
                midpointCache[key] = idx;
                return idx;
            };

            for (const auto& face : faces) {
                idx_t a = getMidpoint(face[0], face[1]);
                idx_t b = getMidpoint(face[1], face[2]);
                idx_t c = getMidpoint(face[2], face[0]);

                newFaces.push_back({face[0], a, c});
                newFaces.push_back({face[1], b, a});
                newFaces.push_back({face[2], c, b});
                newFaces.push_back({a, b, c});
            }
            faces = std::move(newFaces);
        }

        // Build triangles (normals pointing inward for a "room")
        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            // Reverse winding to make normals point inward
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Mersenne Twister -- bit identical replacement for std::mt19937
// ============================================================================
//
// MT19937 is a strictly sequential recurrence, so the host has to walk the whole
// stream.  It only writes out one state snapshot every MT_SNAP_BLOCKS blocks
// though; the device blows those back up into the full word stream in parallel.
// Tempering and the conversion to [0,1) floats happen on the GPU as well, so the
// only data crossing PCIe is 1/MT_SNAP_BLOCKS of the random numbers.

constexpr int MT_N = 624;              // Mersenne Twister state size
constexpr int MT_M = 397;
constexpr uint32_t MT_MATRIX_A = 0x9908b0dfu;
constexpr uint32_t MT_UPPER = 0x80000000u;
constexpr uint32_t MT_LOWER = 0x7fffffffu;

// Number of 624 word blocks regenerated on the device from a single state
// snapshot.  Only snapshots cross PCIe, which cuts the random number traffic by
// this factor.
constexpr int MT_SNAP_BLOCKS = 64;
constexpr size_t MT_SEG_WORDS = static_cast<size_t>(MT_N) * MT_SNAP_BLOCKS;

__host__ __device__ __forceinline__ uint32_t mtMix(uint32_t a, uint32_t b) {
    uint32_t y = (a & MT_UPPER) | (b & MT_LOWER);
    return (y >> 1) ^ ((y & 1u) ? MT_MATRIX_A : 0u);
}

// Host side generator: reproduces std::mt19937 exactly, but only materializes
// the state every MT_SNAP_BLOCKS blocks.  The twisting itself stays in a 2.5 kB
// working set, so it runs entirely out of L1.
class MtSnapshots {
    uint32_t st[MT_N];        // state that produces block `blockIdx`
    long long blockIdx = 0;
    uint32_t lastSnap[MT_N];
    long long lastSnapSeg = -1;

    void twist() {
        int i;
        for (i = 0; i < MT_N - MT_M; ++i) st[i] = st[i + MT_M] ^ mtMix(st[i], st[i + 1]);
        for (; i < MT_N - 1; ++i)         st[i] = st[i - (MT_N - MT_M)] ^ mtMix(st[i], st[i + 1]);
        st[MT_N - 1] = st[MT_M - 1] ^ mtMix(st[MT_N - 1], st[0]);
        ++blockIdx;
    }

public:
    explicit MtSnapshots(uint32_t seed = 42) {
        st[0] = seed;
        for (int i = 1; i < MT_N; ++i) {
            st[i] = 1812433253u * (st[i - 1] ^ (st[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
    }

    // Write the snapshots of segments [s0, s1) as MT_N words each.  Segments are
    // requested in non-decreasing order; the previous one may be repeated when a
    // chunk boundary falls inside a segment.
    void emit(long long s0, long long s1, uint32_t* out) {
        for (long long seg = s0; seg < s1; ++seg, out += MT_N) {
            if (seg == lastSnapSeg) {
                std::memcpy(out, lastSnap, MT_N * sizeof(uint32_t));
                continue;
            }
            const long long target = seg * MT_SNAP_BLOCKS;
            while (blockIdx < target) twist();
            std::memcpy(out, st, MT_N * sizeof(uint32_t));
            std::memcpy(lastSnap, st, MT_N * sizeof(uint32_t));
            lastSnapSeg = seg;
        }
    }
};

// Expand one state snapshot per block back into MT_SEG_WORDS stream words.
__global__ __launch_bounds__(256) void mtExpandKernel(const uint32_t* __restrict__ snaps,
                                                      uint32_t* __restrict__ out) {
    __shared__ uint32_t bufA[MT_N];
    __shared__ uint32_t bufB[MT_N];

    const int seg = blockIdx.x;
    const int tid = threadIdx.x;
    uint32_t* cur = bufA;
    uint32_t* nxt = bufB;

    for (int i = tid; i < MT_N; i += 256) cur[i] = snaps[static_cast<size_t>(seg) * MT_N + i];
    __syncthreads();

    uint32_t* dst = out + static_cast<size_t>(seg) * MT_SEG_WORDS;
    for (int k = 0; k < MT_SNAP_BLOCKS; ++k) {
        for (int i = tid; i < MT_N - MT_M; i += 256)
            nxt[i] = cur[i + MT_M] ^ mtMix(cur[i], cur[i + 1]);
        __syncthreads();
        for (int i = tid + (MT_N - MT_M); i < 2 * (MT_N - MT_M); i += 256)
            nxt[i] = nxt[i - (MT_N - MT_M)] ^ mtMix(cur[i], cur[i + 1]);
        __syncthreads();
        for (int i = tid + 2 * (MT_N - MT_M); i < MT_N - 1; i += 256)
            nxt[i] = nxt[i - (MT_N - MT_M)] ^ mtMix(cur[i], cur[i + 1]);
        if (tid == 0) nxt[MT_N - 1] = nxt[MT_M - 1] ^ mtMix(cur[MT_N - 1], nxt[0]);
        __syncthreads();

        for (int i = tid; i < MT_N; i += 256) dst[i] = nxt[i];
        dst += MT_N;

        uint32_t* tmp = cur; cur = nxt; nxt = tmp;
        __syncthreads();
    }
}

// ============================================================================
// Device side scene representation
// ============================================================================

struct DevScene {
    // 3 float4 per triangle: {a, b-a, c-a}; interleaved so one triangle test
    // touches a single 48 byte run.
    const float4* __restrict__ triGeom;
    const float4* __restrict__ triN;    // xyz: face normal
    // 2 float4 per octree node: {center, half extent}.  Nodes are laid out
    // breadth first, so the (up to 8) children of a node are contiguous.
    const float4* __restrict__ nodeBox;
    // x: first child / first triangle reference,
    // y: >0 -> leaf triangle count, <=0 -> negated 8 bit child mask
    const int2* __restrict__ nodeInfo;
    const int* __restrict__ nodeTriList;
};

__device__ __forceinline__ float3 xyz(const float4& v) { return make_float3(v.x, v.y, v.z); }
__device__ __forceinline__ float3 operator+(const float3& a, const float3& b) {
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}
__device__ __forceinline__ float3 operator-(const float3& a, const float3& b) {
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}
__device__ __forceinline__ float3 operator*(const float3& a, float s) {
    return make_float3(a.x * s, a.y * s, a.z * s);
}
__device__ __forceinline__ float3 operator-(const float3& a) {
    return make_float3(-a.x, -a.y, -a.z);
}
__device__ __forceinline__ float dot3(const float3& a, const float3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ float3 cross3(const float3& a, const float3& b) {
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

// std::mt19937 tempering
__device__ __forceinline__ uint32_t mtTemper(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    return y;
}

// libstdc++ std::uniform_real_distribution<float>(0,1) over std::mt19937
__device__ __forceinline__ float mtCanonical(uint32_t raw) {
    float r = __uint2float_rn(mtTemper(raw)) * 2.3283064365386963e-10f;  // 2^-32
    // std::generate_canonical clamps to nextafter(1, 0)
    return r >= 1.0f ? 0x1.fffffep-1f : r;
}

// Möller-Trumbore, identical arithmetic to the sequential version
__device__ __forceinline__ float rayTriangleIntersect(const float3& orig, const float3& dir,
                                                      const float3& v0, const float3& e1,
                                                      const float3& e2) {
    float3 pvec = cross3(dir, e2);
    float det = dot3(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX_DEV;

    float invDet = 1.0f / det;
    float3 tvec = orig - v0;
    float u = dot3(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX_DEV;

    float3 qvec = cross3(tvec, e1);
    float v = dot3(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX_DEV;

    return dot3(e2, qvec) * invDet;
}

// Segment/box separating axis test.  `mid`, `d` and `ad` are the per-ray
// constants (segment midpoint, half vector and its absolute value) that the
// reference recomputes for every node.
__device__ __forceinline__ bool rayIntersectsBox(const float3& mid, const float3& d, const float3& ad,
                                                 const float3& center, const float3& halfExtent) {
    float3 c = mid - center;

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

constexpr int OCTREE_STACK_SIZE = 64;

// Is the segment from `from` to `to` blocked by a triangle other than src/dst?
// The traversal visits exactly the leaves the recursive reference visits; since
// the result is a pure OR over triangles the visiting order is irrelevant.
__device__ bool isRayBlocked(const DevScene& s, const float3& from, const float3& to,
                             int srcTriIdx, int dstTriIdx) {
    float3 dir = to - from;
    float rayLen = sqrtf(dot3(dir, dir));
    if (rayLen < EPSILON) return true;
    float3 dirNorm = make_float3(dir.x / rayLen, dir.y / rayLen, dir.z / rayLen);

    const float3 d = dir * 0.5f;
    const float3 mid = from + d;
    const float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    int stack[OCTREE_STACK_SIZE];
    int sp = 0;
    int node = 0;  // root is entered without a bounding box test

    for (;;) {
        int2 inf = s.nodeInfo[node];
        if (inf.y > 0) {
            for (int k = 0; k < inf.y; ++k) {
                int idx = s.nodeTriList[inf.x + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const float4* g = s.triGeom + 3 * idx;
                float dist = rayTriangleIntersect(from, dirNorm, xyz(g[0]), xyz(g[1]), xyz(g[2]));
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            int mask = -inf.y;
            int cid = inf.x;
            int next = -1;
            while (mask) {
                mask &= mask - 1;  // consume the lowest set bit
                const float4* bx = s.nodeBox + 2 * cid;
                if (rayIntersectsBox(mid, d, ad, xyz(bx[0]), xyz(bx[1]))) {
                    if (next < 0) next = cid;       // descend into the first hit
                    else stack[sp++] = cid;         // defer the rest
                }
                ++cid;
            }
            if (next >= 0) { node = next; continue; }
        }
        if (sp == 0) break;
        node = stack[--sp];
    }
    return false;
}

__device__ __forceinline__ float cosPhiDev(const float3& v, const float3& normal) {
    float vNorm = sqrtf(dot3(v, v));
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, dot3(v, normal) / vNorm);
}

// ============================================================================
// Kernels
// ============================================================================

// Tau: ceil(|center_i - center_j| / WAVE_SPEED), one thread per pair.
__global__ void tauKernel(const float4* __restrict__ triCenter, int N, unsigned char* __restrict__ tau) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    int i = blockIdx.y;
    size_t off = static_cast<size_t>(i) * N + j;
    if (i == j) { tau[off] = 0; return; }
    float3 d = xyz(triCenter[i]) - xyz(triCenter[j]);
    float dist = sqrtf(dot3(d, d));
    // The mesh diameter is 20 units, so the delay never exceeds 40 timesteps.
    tau[off] = static_cast<unsigned char>(static_cast<int>(ceilf(dist * INV_WAVE_SPEED)));
}

// Exclusive prefix sum, within one mesh row, of "this pair consumes randomness".
// One block per row of the current chunk.
__global__ void rowRankKernel(const uint32_t* __restrict__ maskBits, int N, int rowBase,
                              uint32_t* __restrict__ rowRank) {
    __shared__ uint32_t sdata[256];
    __shared__ uint32_t running;

    const int local = blockIdx.x;
    const int i = rowBase + local;
    const int tid = threadIdx.x;

    if (tid == 0) running = 0;
    __syncthreads();

    for (int base = 0; base < N; base += 256) {
        int j = base + tid;
        uint32_t m = 0;
        if (j < N) {
            size_t bit = static_cast<size_t>(i) * N + j;
            m = (maskBits[bit >> 5] >> (bit & 31)) & 1u;
        }
        sdata[tid] = m;
        __syncthreads();
        for (int d = 1; d < 256; d <<= 1) {
            uint32_t t = (tid >= d) ? sdata[tid - d] : 0u;
            __syncthreads();
            if (tid >= d) sdata[tid] += t;
            __syncthreads();
        }
        uint32_t incl = sdata[tid];
        uint32_t total = sdata[255];
        if (j < N) rowRank[static_cast<size_t>(local) * N + j] = running + incl - m;
        __syncthreads();
        if (tid == 0) running += total;
        __syncthreads();
    }
}

// Form factors.  NUM_RAYS lanes cooperate on one (i,j) pair, one shadow ray each.
__global__ __launch_bounds__(128) void kijKernel(DevScene s, int N, int rowBase,
                                                 const uint32_t* __restrict__ rowRank,
                                                 const unsigned long long* __restrict__ rowStart,
                                                 const uint32_t* __restrict__ maskBits,
                                                 const uint32_t* __restrict__ rng,
                                                 unsigned long long rngWordBase,
                                                 float* __restrict__ kij) {
    // A pair is handled by NUM_RAYS consecutive lanes: the rays of one pair run
    // between the same two triangles, so they traverse almost identical octree
    // paths and the group stays convergent.
    const int lane = threadIdx.x & (NUM_RAYS - 1);
    const int group = threadIdx.x / NUM_RAYS;
    const int j = blockIdx.x * (blockDim.x / NUM_RAYS) + group;
    const int local = blockIdx.y;
    const int i = rowBase + local;

    const size_t off = static_cast<size_t>(i) * N + j;          // global pair index
    const size_t outOff = static_cast<size_t>(local) * N + j;    // index inside this chunk
    // Group uniform predicates -- no thread leaves the warp before the shuffles.
    const bool live = (j < N);
    const uint32_t m = live ? ((maskBits[off >> 5] >> (off & 31)) & 1u) : 0u;

    float contrib = ZERO;
    if (m) {
        unsigned long long rank = rowStart[i] + rowRank[static_cast<size_t>(local) * N + j];
        const size_t w = static_cast<size_t>(rank * DRAWS_PER_PAIR - rngWordBase);
        uint4 q = reinterpret_cast<const uint4*>(rng + w)[lane];

        const float4* gI = s.triGeom + 3 * i;
        const float4* gJ = s.triGeom + 3 * j;
        const float3 aI = xyz(gI[0]), abI = xyz(gI[1]), acI = xyz(gI[2]);
        const float3 aJ = xyz(gJ[0]), abJ = xyz(gJ[1]), acJ = xyz(gJ[2]);
        const float3 nI = xyz(s.triN[i]), nJ = xyz(s.triN[j]);

        float u1 = mtCanonical(q.x), v1 = mtCanonical(q.y);
        float u2 = mtCanonical(q.z), v2 = mtCanonical(q.w);
        if (u1 + v1 > 1.0f) { u1 = 1.0f - u1; v1 = 1.0f - v1; }
        if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }

        float3 pI = aI + abI * u1 + acI * v1;
        float3 pJ = aJ + abJ * u2 + acJ * v2;

        float3 v = pJ - pI;
        float distSqr = dot3(v, v);
        // Cheap rejections first; they are order independent w.r.t. the
        // visibility test, so the accumulated value is unchanged.
        if (distSqr >= EPSILON) {
            float cosPhiI = cosPhiDev(v, nI);
            float cosPhiJ = cosPhiDev(-v, nJ);
            if (cosPhiI > ZERO && cosPhiJ > ZERO && !isRayBlocked(s, pI, pJ, i, j)) {
                contrib = (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
        }
    }

    // Reduce in ray order; skipped rays contribute an exact +0.0f, so this
    // matches the sequential accumulation bit for bit.
    float acc = ZERO;
    #pragma unroll
    for (int r = 0; r < NUM_RAYS; ++r) {
        acc += __shfl_sync(0xffffffffu, contrib, r, NUM_RAYS);
    }

    if (live && lane == 0) kij[outOff] = acc * INV_NUM_RAYS;
}

// One timestep of the radiosity update; one thread per receiver triangle.
// The j loop keeps the sequential summation order (and reads the transposed
// weights so consecutive threads touch consecutive addresses).  Loads for eight
// neighbours are issued together to give the few resident warps enough
// memory level parallelism.
__global__ void radiosityKernel(int N, int t,
                                const float* __restrict__ weightT,
                                const unsigned char* __restrict__ tauT,
                                const float* __restrict__ rho,
                                const float* __restrict__ radE,
                                float* __restrict__ radB) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    constexpr int U = 8;
    float sumB = ZERO;
    for (int j0 = 0; j0 < N; j0 += U) {
        float w[U];
        int tv[U];
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            int j = j0 + u;
            size_t sym = static_cast<size_t>(j) * N + i;
            w[u] = (j < N) ? weightT[sym] : ZERO;
            tv[u] = (j < N) ? tauT[sym] : 0;
        }
        #pragma unroll
        for (int u = 0; u < U; ++u) {
            int j = j0 + u;
            if (j >= N || j == i) continue;
            if (t < tv[u]) continue;
            if (w[u] <= ZERO) continue;   // equivalent to the reference's kij <= 0
            float radJ = radB[static_cast<size_t>(t - tv[u]) * N + j];
            if (radJ <= ZERO) continue;
            sumB += w[u] * radJ;
        }
    }

    size_t o = static_cast<size_t>(t) * N + i;
    radB[o] = rho[i] * sumB + radE[o];
}

// Transposed reflection weights: weightT[j*N+i] = min(Kij[i][j] * area[j], 1).
__global__ void weightsKernel(const float* __restrict__ kij, const float* __restrict__ areas,
                              int N, float* __restrict__ weightT) {
    __shared__ float tile[32][33];
    int x = blockIdx.x * 32 + threadIdx.x;          // j
    int y = blockIdx.y * 32 + threadIdx.y;          // i
    for (int k = 0; k < 32; k += 8) {
        if (x < N && (y + k) < N) tile[threadIdx.y + k][threadIdx.x] = kij[static_cast<size_t>(y + k) * N + x];
    }
    __syncthreads();
    int oi = blockIdx.y * 32 + threadIdx.x;         // i
    int oj = blockIdx.x * 32 + threadIdx.y;         // j
    for (int k = 0; k < 32; k += 8) {
        if (oi < N && (oj + k) < N) {
            float kv = tile[threadIdx.x][threadIdx.y + k];
            weightT[static_cast<size_t>(oj + k) * N + oi] = fminf(kv * areas[oj + k], ONE);
        }
    }
}

// Cross correlation of every triangle's response against the source response.
__global__ void correlationKernel(int N, int T, int sourceIndex,
                                  const float* __restrict__ radB,
                                  float* __restrict__ corr) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y;
    if (i >= N) return;

    float sum = ZERO;
    for (int tt = t; tt < T; ++tt) {
        float pB = radB[static_cast<size_t>(tt) * N + i];
        float pS = radB[static_cast<size_t>(tt - t) * N + sourceIndex];
        sum += pS * pB;
    }
    corr[static_cast<size_t>(t) * N + i] = sum;
}

// Argmax over lags, first maximum wins (matches the sequential ">" test).
__global__ void argmaxKernel(int N, int T, const float* __restrict__ corr,
                             float* __restrict__ distances) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float maxCorr = ZERO;
    int bestT = 0;
    for (int t = 0; t < T; ++t) {
        float sum = corr[static_cast<size_t>(t) * N + i];
        if (sum > maxCorr) { maxCorr = sum; bestT = t; }
    }
    distances[i] = WAVE_SPEED * static_cast<float>(bestT);
}

// ============================================================================
// Flattened octree
// ============================================================================

struct FlatOctree {
    std::vector<float4> box;    // 2 entries per node: center, half extent
    std::vector<int2> info;
    std::vector<int> triList;
    int maxDepth = 0;
    size_t numNodes() const { return info.size(); }
};

// Breadth first flattening so that siblings -- which are always tested together
// -- end up in consecutive memory.
static void flattenOctree(const Octree* root, FlatOctree& f) {
    struct Item { const Octree* node; int depth; };
    std::vector<Item> queue;
    queue.push_back({root, 0});
    f.info.resize(1);
    f.box.resize(2);
    f.box[0] = make_float4(root->center.x, root->center.y, root->center.z, 0.0f);
    f.box[1] = make_float4(root->halfExtent.x, root->halfExtent.y, root->halfExtent.z, 0.0f);

    for (size_t k = 0; k < queue.size(); ++k) {
        const Octree* n = queue[k].node;
        f.maxDepth = std::max(f.maxDepth, queue[k].depth);

        if (!n->triangleIndices.empty()) {
            f.info[k] = make_int2(static_cast<int>(f.triList.size()),
                                  static_cast<int>(n->triangleIndices.size()));
            for (size_t idx : n->triangleIndices) f.triList.push_back(static_cast<int>(idx));
            continue;
        }

        const int first = static_cast<int>(queue.size());
        int mask = 0;
        for (int c = 0; c < 8; ++c) {
            if (!n->children[c]) continue;
            mask |= 1 << c;
            queue.push_back({n->children[c].get(), queue[k].depth + 1});
        }
        f.info.resize(queue.size());
        f.box.resize(2 * queue.size());
        for (size_t q = static_cast<size_t>(first); q < queue.size(); ++q) {
            const Octree* ch = queue[q].node;
            f.box[2 * q + 0] = make_float4(ch->center.x, ch->center.y, ch->center.z, 0.0f);
            f.box[2 * q + 1] = make_float4(ch->halfExtent.x, ch->halfExtent.y, ch->halfExtent.z, 0.0f);
        }
        f.info[k] = make_int2(first, -mask);
    }
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// Per GPU replica of the read-only ray tracing inputs.
struct GpuWorker {
    int dev = 0;
    DevScene scene{};
    float4 *dTriGeom = nullptr, *dTriN = nullptr, *dNodeBox = nullptr;
    int2 *dNodeInfo = nullptr;
    int *dNodeTriList = nullptr;
    uint32_t *dMaskBits = nullptr;
    unsigned long long *dRowStart = nullptr;
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flat;                // GPU friendly copy of the octree

    size_t sourceIndex;

    // --- device resident state -------------------------------------------
    // One worker per GPU; the ray tracing phase is spread over all of them,
    // everything else runs on worker 0's device.
    std::vector<GpuWorker> workers;
    float4 *dTriCenter = nullptr;
    float *dKij = nullptr, *dWeightT = nullptr;
    unsigned char *dTau = nullptr;
    float *dAreas = nullptr, *dRho = nullptr, *dRadE = nullptr, *dRadB = nullptr;
    float *dCorr = nullptr, *dDistances = nullptr;
    bool kijHostPinned = false;

    // host side helpers for the randomness stream
    std::vector<uint32_t> maskBits;
    std::vector<unsigned long long> rowStart;   // global rank of the first pair of each row

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    // Generate mesh
    IcosphereMesh mesh(subdivisions, 10.0f);  // Radius 10 units
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    // Build octree for spatial acceleration
    printf("Building octree...\n");
    state.octree.build(state.triangles);
    flattenOctree(&state.octree, state.flat);
    if (state.flat.maxDepth * 7 + 8 > OCTREE_STACK_SIZE) {
        fprintf(stderr, "Octree too deep for the device traversal stack (%d)\n", state.flat.maxDepth);
        exit(1);
    }

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // ---------------------------------------------------------------- device
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    const size_t numNodes = state.flat.numNodes();

    std::vector<float4> hGeom(3 * N), hN(N), hC(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& tr = state.triangles[i];
        Vec3 ab = tr.b - tr.a;
        Vec3 ac = tr.c - tr.a;
        Vec3 nn = tr.normal();
        Vec3 cc = tr.center();
        hGeom[3 * i + 0] = make_float4(tr.a.x, tr.a.y, tr.a.z, 0.0f);
        hGeom[3 * i + 1] = make_float4(ab.x, ab.y, ab.z, 0.0f);
        hGeom[3 * i + 2] = make_float4(ac.x, ac.y, ac.z, 0.0f);
        hN[i] = make_float4(nn.x, nn.y, nn.z, 0.0f);
        hC[i] = make_float4(cc.x, cc.y, cc.z, 0.0f);
    }

    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount < 1) {
        fprintf(stderr, "No CUDA device available\n");
        exit(1);
    }
    // Several GPUs only pay off once ray tracing dominates their setup cost.
    int numDev = (N >= 4096) ? deviceCount : 1;
    printf("Uploading scene to %d GPU(s) (%zu octree nodes, %zu leaf references)...\n",
           numDev, numNodes, state.flat.triList.size());

    state.workers.resize(numDev);
    // Context creation costs a few hundred milliseconds per GPU, so bring the
    // devices up (and fill them) concurrently.
    auto bringUp = [&](int d) {
        GpuWorker& w = state.workers[d];
        w.dev = d;
        CUDA_CHECK(cudaSetDevice(d));
        w.dTriGeom = devAlloc<float4>(3 * N);
        w.dTriN = devAlloc<float4>(N);
        w.dNodeBox = devAlloc<float4>(2 * numNodes);
        w.dNodeInfo = devAlloc<int2>(numNodes);
        w.dNodeTriList = devAlloc<int>(std::max<size_t>(1, state.flat.triList.size()));
        CUDA_CHECK(cudaMemcpy(w.dTriGeom, hGeom.data(), 3 * N * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w.dTriN, hN.data(), N * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w.dNodeBox, state.flat.box.data(), 2 * numNodes * sizeof(float4), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w.dNodeInfo, state.flat.info.data(), numNodes * sizeof(int2), cudaMemcpyHostToDevice));
        if (!state.flat.triList.empty()) {
            CUDA_CHECK(cudaMemcpy(w.dNodeTriList, state.flat.triList.data(),
                                  state.flat.triList.size() * sizeof(int), cudaMemcpyHostToDevice));
        }
        w.scene.triGeom = w.dTriGeom;
        w.scene.triN = w.dTriN;
        w.scene.nodeBox = w.dNodeBox;
        w.scene.nodeInfo = w.dNodeInfo;
        w.scene.nodeTriList = w.dNodeTriList;
    };
    if (numDev == 1) {
        bringUp(0);
    } else {
        std::vector<std::thread> up;
        for (int d = 0; d < numDev; ++d) up.emplace_back(bringUp, d);
        for (auto& th : up) th.join();
    }

    CUDA_CHECK(cudaSetDevice(0));
    state.dTriCenter = devAlloc<float4>(N);
    CUDA_CHECK(cudaMemcpy(state.dTriCenter, hC.data(), N * sizeof(float4), cudaMemcpyHostToDevice));

    // Form factor slices come back through host memory when several GPUs
    // cooperate; pinning makes those copies asynchronous.
    if (numDev > 1) {
        state.kijHostPinned =
            cudaHostRegister(state.kij.data(), N * N * sizeof(val_t), cudaHostRegisterPortable) == cudaSuccess;
        cudaGetLastError();
    }

    state.dKij = devAlloc<float>(N * N);
    state.dWeightT = devAlloc<float>(N * N);
    state.dTau = devAlloc<unsigned char>(N * N);
    state.dAreas = devAlloc<float>(N);
    state.dRho = devAlloc<float>(N);
    state.dRadE = devAlloc<float>(T * N);
    state.dRadB = devAlloc<float>(T * N);
    state.dCorr = devAlloc<float>(T * N);
    state.dDistances = devAlloc<float>(N);

    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRadE, state.radE.data(), T * N * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, T * N * sizeof(float)));
}

void releaseSimulation(SimulationState& state) {
    for (GpuWorker& w : state.workers) {
        cudaSetDevice(w.dev);
        for (void* p : {(void*)w.dTriGeom, (void*)w.dTriN, (void*)w.dNodeBox, (void*)w.dNodeInfo,
                        (void*)w.dNodeTriList, (void*)w.dMaskBits, (void*)w.dRowStart}) {
            if (p) cudaFree(p);
        }
    }
    cudaSetDevice(0);
    for (void* p : {(void*)state.dTriCenter, (void*)state.dKij, (void*)state.dWeightT,
                    (void*)state.dTau, (void*)state.dAreas, (void*)state.dRho,
                    (void*)state.dRadE, (void*)state.dRadB, (void*)state.dCorr,
                    (void*)state.dDistances}) {
        if (p) cudaFree(p);
    }
    if (state.kijHostPinned) cudaHostUnregister(state.kij.data());
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int N = static_cast<int>(state.numTriangles);
    dim3 block(256);
    dim3 grid((N + 255) / 256, N);
    tauKernel<<<grid, block>>>(state.dTriCenter, N, state.dTau);
    CUDA_CHECK(cudaGetLastError());
}

// Build the "pair consumes randomness" bit mask exactly as the sequential code
// decides it (same expression, same rounding), so the random stream offsets are
// guaranteed to line up.
static void buildRandomnessMask(SimulationState& state) {
    const size_t N = state.numTriangles;
    state.maskBits.assign((N * N + 31) / 32, 0u);
    state.rowStart.assign(N + 1, 0ull);
    std::vector<uint32_t> rowCount(N, 0u);

    uint32_t* bits = state.maskBits.data();
    const Triangle* tris = state.triangles.data();

    // The cull predicate below must produce exactly the same decision as the
    // sequential reference: it selects which pairs advance the shared random
    // stream, so a single flipped bit would desynchronize everything after it.
    // Hence the expression is kept verbatim and only the row range is split.
    auto worker = [&](size_t iBegin, size_t iEnd) {
        for (size_t i = iBegin; i < iEnd; ++i) {
            const Vec3 nI = tris[i].normal();
            uint32_t count = 0;
            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;
                if (nI.dot(tris[j].normal()) > 0.99f) continue;
                size_t bit = i * N + j;
                bits[bit >> 5] |= 1u << (bit & 31);
                ++count;
            }
            rowCount[i] = count;
        }
    };

    // Rows may only be split across threads when a row boundary is also a word
    // boundary, otherwise two threads would OR into the same word.
    unsigned nThreads = std::thread::hardware_concurrency();
    if (nThreads == 0) nThreads = 1;
    nThreads = std::min<unsigned>(nThreads, 64);
    if (N % 32 != 0 || N < 2 * nThreads) nThreads = 1;

    if (nThreads == 1) {
        worker(0, N);
    } else {
        std::vector<std::thread> pool;
        pool.reserve(nThreads);
        const size_t per = (N + nThreads - 1) / nThreads;
        for (unsigned t = 0; t < nThreads; ++t) {
            size_t b = t * per, e = std::min(N, b + per);
            if (b >= e) break;
            pool.emplace_back(worker, b, e);
        }
        for (auto& th : pool) th.join();
    }

    unsigned long long running = 0;
    for (size_t i = 0; i < N; ++i) {
        state.rowStart[i] = running;
        running += rowCount[i];
    }
    state.rowStart[N] = running;
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const int N = static_cast<int>(state.numTriangles);
    const size_t Nz = state.numTriangles;
    const int numDev = static_cast<int>(state.workers.size());

    buildRandomnessMask(state);

    for (GpuWorker& w : state.workers) {
        CUDA_CHECK(cudaSetDevice(w.dev));
        w.dMaskBits = devAlloc<uint32_t>(state.maskBits.size());
        w.dRowStart = devAlloc<unsigned long long>(Nz);
        CUDA_CHECK(cudaMemcpy(w.dMaskBits, state.maskBits.data(),
                              state.maskBits.size() * sizeof(uint32_t), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(w.dRowStart, state.rowStart.data(),
                              Nz * sizeof(unsigned long long), cudaMemcpyHostToDevice));
    }

    // Chunk the mesh rows so the random stream for one chunk stays around 64 MB
    // and every GPU gets several chunks to pipeline.
    const size_t wordsPerRow = Nz * DRAWS_PER_PAIR;
    size_t rowsPerChunk = std::max<size_t>(1, (16u << 20) / std::max<size_t>(1, wordsPerRow));
    rowsPerChunk = std::min(rowsPerChunk, std::max<size_t>(1, Nz / (8 * static_cast<size_t>(numDev))));
    rowsPerChunk = std::min(rowsPerChunk, Nz);
    const size_t chunkWords = rowsPerChunk * wordsPerRow;

    // Random words per chunk, rounded up to whole snapshot segments.
    const size_t maxSegs = chunkWords / MT_SEG_WORDS + 2;
    const size_t rngWords = maxSegs * MT_SEG_WORDS;

    constexpr int NBUF = 2;
    struct Slot {
        uint32_t* hostSnap = nullptr;
        uint32_t* devSnap = nullptr;
        uint32_t* devRng = nullptr;
        uint32_t* devRowRank = nullptr;
        float* devKij = nullptr;      // only used when several GPUs cooperate
        cudaStream_t stream{};
        cudaEvent_t copyDone{};
    };
    std::vector<Slot> slots(static_cast<size_t>(numDev) * NBUF);

    for (int d = 0; d < numDev; ++d) {
        CUDA_CHECK(cudaSetDevice(state.workers[d].dev));
        for (int b = 0; b < NBUF; ++b) {
            Slot& sl = slots[d * NBUF + b];
            CUDA_CHECK(cudaHostAlloc(&sl.hostSnap, maxSegs * MT_N * sizeof(uint32_t), cudaHostAllocDefault));
            sl.devSnap = devAlloc<uint32_t>(maxSegs * MT_N);
            sl.devRng = devAlloc<uint32_t>(rngWords);
            sl.devRowRank = devAlloc<uint32_t>(rowsPerChunk * Nz);
            if (numDev > 1) sl.devKij = devAlloc<float>(rowsPerChunk * Nz);
            CUDA_CHECK(cudaStreamCreate(&sl.stream));
            CUDA_CHECK(cudaEventCreateWithFlags(&sl.copyDone, cudaEventDisableTiming));
        }
    }

    MtSnapshots mt(42);

    dim3 block(128);
    const unsigned pairsPerBlock = block.x / NUM_RAYS;
    size_t chunkIdx = 0;
    for (size_t rowBase = 0; rowBase < Nz; rowBase += rowsPerChunk, ++chunkIdx) {
        const size_t rows = std::min(rowsPerChunk, Nz - rowBase);
        const int d = static_cast<int>(chunkIdx % static_cast<size_t>(numDev));
        const int b = static_cast<int>((chunkIdx / static_cast<size_t>(numDev)) % NBUF);
        GpuWorker& w = state.workers[d];
        Slot& sl = slots[d * NBUF + b];

        const unsigned long long w0 = state.rowStart[rowBase] * DRAWS_PER_PAIR;
        const unsigned long long w1 = state.rowStart[rowBase + rows] * DRAWS_PER_PAIR;
        const long long s0 = static_cast<long long>(w0 / MT_SEG_WORDS);
        const long long s1 = static_cast<long long>((w1 + MT_SEG_WORDS - 1) / MT_SEG_WORDS);
        const unsigned numSegs = static_cast<unsigned>(s1 - s0);

        CUDA_CHECK(cudaSetDevice(w.dev));

        // The pinned buffer can be refilled once its previous upload finished.
        if (chunkIdx >= static_cast<size_t>(numDev) * NBUF) CUDA_CHECK(cudaEventSynchronize(sl.copyDone));

        if (numSegs > 0) {
            mt.emit(s0, s1, sl.hostSnap);
            CUDA_CHECK(cudaMemcpyAsync(sl.devSnap, sl.hostSnap, numSegs * MT_N * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, sl.stream));
            CUDA_CHECK(cudaEventRecord(sl.copyDone, sl.stream));
            mtExpandKernel<<<numSegs, 256, 0, sl.stream>>>(sl.devSnap, sl.devRng);
        }

        rowRankKernel<<<static_cast<unsigned>(rows), 256, 0, sl.stream>>>(
            w.dMaskBits, N, static_cast<int>(rowBase), sl.devRowRank);

        float* out = (numDev > 1) ? sl.devKij : (state.dKij + rowBase * Nz);
        dim3 grid((N + pairsPerBlock - 1) / pairsPerBlock, static_cast<unsigned>(rows));
        kijKernel<<<grid, block, 0, sl.stream>>>(
            w.scene, N, static_cast<int>(rowBase), sl.devRowRank, w.dRowStart, w.dMaskBits,
            sl.devRng, static_cast<unsigned long long>(s0) * MT_SEG_WORDS, out);
        CUDA_CHECK(cudaGetLastError());

        if (numDev > 1) {
            CUDA_CHECK(cudaMemcpyAsync(state.kij.data() + rowBase * Nz, sl.devKij,
                                       rows * Nz * sizeof(float), cudaMemcpyDeviceToHost, sl.stream));
        }

        for (size_t i = rowBase; i < rowBase + rows; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == Nz) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, Nz);
            }
        }
    }

    for (int d = 0; d < numDev; ++d) {
        CUDA_CHECK(cudaSetDevice(state.workers[d].dev));
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    for (int d = 0; d < numDev; ++d) {
        CUDA_CHECK(cudaSetDevice(state.workers[d].dev));
        for (int b = 0; b < NBUF; ++b) {
            Slot& sl = slots[d * NBUF + b];
            CUDA_CHECK(cudaFreeHost(sl.hostSnap));
            CUDA_CHECK(cudaFree(sl.devSnap));
            CUDA_CHECK(cudaFree(sl.devRng));
            CUDA_CHECK(cudaFree(sl.devRowRank));
            if (sl.devKij) CUDA_CHECK(cudaFree(sl.devKij));
            CUDA_CHECK(cudaStreamDestroy(sl.stream));
            CUDA_CHECK(cudaEventDestroy(sl.copyDone));
        }
    }

    CUDA_CHECK(cudaSetDevice(0));
    if (numDev > 1) {
        CUDA_CHECK(cudaMemcpy(state.dKij, state.kij.data(), Nz * Nz * sizeof(float),
                              cudaMemcpyHostToDevice));
    }

    // The radiosity kernel reads the weights column-major for coalescing.
    dim3 tblock(32, 8);
    dim3 tgrid((N + 31) / 32, (N + 31) / 32);
    weightsKernel<<<tgrid, tblock>>>(state.dKij, state.dAreas, N, state.dWeightT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x);

    for (int t = 0; t < T; ++t) {
        // Tau is symmetric, so reading it transposed is exact.
        radiosityKernel<<<grid, block>>>(N, t, state.dWeightT, state.dTau,
                                         state.dRho, state.dRadE, state.dRadB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    dim3 block(128);
    dim3 grid((N + block.x - 1) / block.x, static_cast<unsigned>(T));
    correlationKernel<<<grid, block>>>(N, T, static_cast<int>(state.sourceIndex),
                                       state.dRadB, state.dCorr);
    CUDA_CHECK(cudaGetLastError());

    argmaxKernel<<<dim3((N + block.x - 1) / block.x), block>>>(N, T, state.dCorr, state.dDistances);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances, N * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %zu: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %zu: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    int receivedEnergy = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.radB[state.idxTN(t, i)] > EPSILON) {
                receivedEnergy++;
                break;
            }
        }
    }

    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    int nonZeroKij = 0;
    for (size_t i = 0; i < state.numTriangles * state.numTriangles; ++i) {
        if (state.kij[i] > EPSILON) nonZeroKij++;
    }
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, state.numTriangles * state.numTriangles,
           100.0f * nonZeroKij / static_cast<val_t>(state.numTriangles * state.numTriangles));

    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) {
        return false;
    }

    printf("  Validation: PASSED\n");
    return true;
}

// Pull the arrays that validation inspects back from the device.
static void fetchValidationData(SimulationState& state) {
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;
    CUDA_CHECK(cudaMemcpy(state.kij.data(), state.dKij, N * N * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, T * N * sizeof(float), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.numTriangles; ++i) {
        const uint32_t* ptr = reinterpret_cast<const uint32_t*>(&state.distances[i]);
        hash ^= (static_cast<uint64_t>(*ptr) + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

// ============================================================================
// Helper: Compute subdivision level from target triangle count
// ============================================================================

int getSubdivisionsForTriangleCount(int targetTriangles) {
    // Icosphere: 20 triangles initially, 4x per subdivision
    // subdivisions: 0->20, 1->80, 2->320, 3->1280, 4->5120, 5->20480
    int subdivisions = 0;
    int triangles = 20;
    while (triangles < targetTriangles && subdivisions < 6) {
        subdivisions++;
        triangles *= 4;
    }
    return subdivisions;
}

// ============================================================================
// Main
// ============================================================================

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Target number of triangles (default: 320)\n");
    printf("               Actual count will be rounded to nearest icosphere level:\n");
    printf("               20, 80, 320, 1280, 5120, 20480\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            targetTriangles = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-t") == 0 && i + 1 < argc) {
            timesteps = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-s") == 0 && i + 1 < argc) {
            sourceIdx = atoi(argv[++i]);
        } else if (strcmp(argv[i], "-r") == 0 && i + 1 < argc) {
            reflectivity = static_cast<val_t>(atof(argv[++i]));
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-o") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            printUsage(argv[0]);
            return 0;
        } else {
            printf("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    printf("Room Response Simulation Benchmark\n");
    printf("===================================\n");
    printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    printf("Timesteps: %d\n", timesteps);
    printf("Source triangle: %d\n", sourceIdx);
    printf("Reflectivity: %.2f\n", reflectivity);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("\n");

    CUDA_CHECK(cudaFree(nullptr));  // pay the CUDA context creation before timing

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Simulation
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    printf("Simulation time: %ld ms\n", simDuration);
    printf("\n");

    // Distance computation
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    printf("Distance computation time: %ld ms\n", distDuration);
    printf("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    printf("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    printf("\nPerformance:\n");
    printf("  Triangles: %zu\n", n);
    printf("  Timesteps: %zu\n", t);
    printf("  Form factor computations: %.2e\n", kijOps);
    printf("  Simulation operations: %.2e\n", simOps);
    printf("  Distance computations: %.2e\n", distOps);
    printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    printf("  Result hash: %016lX\n", hash);
    printf("\n");

    // Print results for external validation
    if (printResults) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    if (validate) {
        fetchValidationData(state);
        if (!validateResults(state)) {
            releaseSimulation(state);
            return 1;
        }
    }

    releaseSimulation(state);
    return 0;
}
