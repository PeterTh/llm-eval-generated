/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified, purely sequential implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
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

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d: %s\n", #call, __FILE__,       \
                    __LINE__, cudaGetErrorString(err_));                          \
            exit(1);                                                              \
        }                                                                         \
    } while (0)

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

// ============================================================================
// Vector and Triangle Types
// ============================================================================

struct Vec3 {
    val_t x, y, z;

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    explicit HD constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    HD val_t norm() const { return sqrtf(squaredNorm()); }
    HD Vec3 normalized() const {
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
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
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
// Octree for Spatial Acceleration
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

public:
    // Check if a ray intersects this node's bounding box
    bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        if (std::abs(c.x) > halfExtent.x + ad.x) return false;
        if (std::abs(c.y) > halfExtent.y + ad.y) return false;
        if (std::abs(c.z) > halfExtent.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + EPSILON) return false;

        return true;
    }

    // Apply a function to all triangles potentially intersecting the ray
    // Returns true if the function returns true for any triangle
    template<typename Func>
    bool applyToTris(const Vec3& p1, const Vec3& p2, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(p1, p2)) {
                if (children[i]->applyToTris(p1, p2, func)) return true;
            }
        }
        return false;
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
// Random Number Generation
// ============================================================================

class RandomGenerator {
    std::mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// Generate a random point inside a triangle using barycentric coordinates
Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
    val_t u = rng.rand();
    val_t v = rng.rand();
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// ============================================================================
// Device-side Octree (flattened) and Random Stream
// ============================================================================

constexpr int FF_BLOCK = 128;            // Threads per form-factor block (= j-tile width)
constexpr int RNG_PER_PAIR = NUM_RAYS * 4;  // 4 uniform draws per ray
constexpr int OCTREE_STACK = 128;
constexpr uint8_t TAU_SKIP = 255;        // Sentinel: never propagates (diagonal)

struct DevNode {
    Vec3 center, half;
    int child[8];
    int triStart, triCount;
};

// Leaf triangle with precomputed edges for Möller-Trumbore
struct LeafTri {
    Vec3 v0; int idx;
    Vec3 e1; int pad0;
    Vec3 e2; int pad1;
};

struct DevOctree {
    std::vector<DevNode> nodes;
    std::vector<LeafTri> tris;
};

static int flattenOctree(const Octree& node, const std::vector<Triangle>& all, DevOctree& out) {
    int id = static_cast<int>(out.nodes.size());
    out.nodes.emplace_back();
    DevNode dn;
    dn.center = node.center;
    dn.half = node.halfExtent;
    for (int c = 0; c < 8; ++c) dn.child[c] = -1;
    dn.triStart = static_cast<int>(out.tris.size());
    dn.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) {
        const Triangle& t = all[idx];
        LeafTri lt;
        lt.v0 = t.a;
        lt.e1 = t.b - t.a;
        lt.e2 = t.c - t.a;
        lt.idx = static_cast<int>(idx);
        lt.pad0 = lt.pad1 = 0;
        out.tris.push_back(lt);
    }
    for (int c = 0; c < 8; ++c) {
        if (node.children[c]) dn.child[c] = flattenOctree(*node.children[c], all, out);
    }
    out.nodes[id] = dn;
    return id;
}

// Sequential MT19937 stream generator (host). Produces the exact raw output
// sequence of std::mt19937(seed) before tempering; tempering is done on the GPU.
class MtRawStream {
    static constexpr int N = 624, M = 397;
    uint32_t mt[N];
    int pos;

    static inline uint32_t mix(uint32_t a, uint32_t b) {
        uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
        return (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu);
    }
    void twist() {
        for (int k = 0; k < N - M; ++k) mt[k] = mt[k + M] ^ mix(mt[k], mt[k + 1]);
        for (int k = N - M; k < N - 1; ++k) mt[k] = mt[k + (M - N)] ^ mix(mt[k], mt[k + 1]);
        mt[N - 1] = mt[M - 1] ^ mix(mt[N - 1], mt[0]);
        pos = 0;
    }
public:
    explicit MtRawStream(uint32_t seed) : pos(N) {
        mt[0] = seed;
        for (int i = 1; i < N; ++i) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
    }
    void fill(uint32_t* out, size_t count) {
        while (count > 0) {
            if (pos >= N) twist();
            size_t n = std::min(count, static_cast<size_t>(N - pos));
            memcpy(out, mt + pos, n * sizeof(uint32_t));
            pos += static_cast<int>(n);
            out += n;
            count -= n;
        }
    }
};

// ============================================================================
// Device Code: Ray-Triangle Intersection, Visibility, Form Factors
// ============================================================================

__device__ __forceinline__ val_t devTemperedUniform(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    // Same as libstdc++ generate_canonical<float,24>: float(y) / 2^32, clamped below 1
    val_t u = __uint2float_rn(y) * 2.3283064365386963e-10f;
    return u >= 1.0f ? 0.99999994f : u;
}

__device__ __forceinline__ Vec3 devRandomPoint(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__device__ __forceinline__ bool devRayIntersectsBox(const DevNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - n.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > n.half.x + ad.x) return false;
    if (fabsf(c.y) > n.half.y + ad.y) return false;
    if (fabsf(c.z) > n.half.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > n.half.y * ad.z + n.half.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > n.half.z * ad.x + n.half.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > n.half.x * ad.y + n.half.y * ad.x + EPSILON) return false;
    return true;
}

__device__ __forceinline__ bool devIsRayBlocked(const Vec3& from, const Vec3& to,
                                                const DevNode* nodes, const LeafTri* tris,
                                                int srcIdx, int dstIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[OCTREE_STACK];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const DevNode& n = nodes[stack[--sp]];
        if (n.triCount > 0) {
            for (int k = 0; k < n.triCount; ++k) {
                const LeafTri& lt = tris[n.triStart + k];
                if (lt.idx == srcIdx || lt.idx == dstIdx) continue;

                Vec3 pvec = dirNorm.cross(lt.e2);
                val_t det = lt.e1.dot(pvec);
                if (fabsf(det) < EPSILON) continue;
                val_t invDet = 1.0f / det;
                Vec3 tvec = from - lt.v0;
                val_t u = tvec.dot(pvec) * invDet;
                if (u < 0.0f || u > 1.0f) continue;
                Vec3 qvec = tvec.cross(lt.e1);
                val_t v = dirNorm.dot(qvec) * invDet;
                if (v < 0.0f || u + v > 1.0f) continue;
                val_t dist = lt.e2.dot(qvec) * invDet;
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 7; c >= 0; --c) {
                int ci = n.child[c];
                if (ci >= 0 && devRayIntersectsBox(nodes[ci], from, to)) stack[sp++] = ci;
            }
        }
    }
    return false;
}

// Culling predicate: triangles facing the same direction do not exchange energy.
// Explicit intrinsics keep the result independent of FMA contraction.
__device__ __forceinline__ bool devCulled(const Triangle& a, const Triangle& b) {
    Vec3 na = a.normal(), nb = b.normal();
    val_t d = __fadd_rn(__fadd_rn(__fmul_rn(na.x, nb.x), __fmul_rn(na.y, nb.y)), __fmul_rn(na.z, nb.z));
    return d > 0.99f;
}

// Per (row, tile) count of pairs that consume random numbers
__global__ void countPairsKernel(const Triangle* tris, int n, int nTiles, uint32_t* tileCnt) {
    int i = blockIdx.y;
    int j = blockIdx.x * FF_BLOCK + threadIdx.x;
    __shared__ uint32_t warpCnt[FF_BLOCK / 32];
    bool active = j < n && j != i && !devCulled(tris[i], tris[j]);
    uint32_t c = __popc(__ballot_sync(0xffffffffu, active));
    if ((threadIdx.x & 31) == 0) warpCnt[threadIdx.x >> 5] = c;
    __syncthreads();
    if (threadIdx.x == 0) {
        uint32_t s = 0;
        for (int w = 0; w < FF_BLOCK / 32; ++w) s += warpCnt[w];
        tileCnt[static_cast<size_t>(i) * nTiles + blockIdx.x] = s;
    }
}

struct FormFactorParams {
    const Triangle* tris;
    const val_t* areas;
    const DevNode* nodes;
    const LeafTri* leafTris;
    const uint32_t* stream;       // raw MT words for this batch
    const uint64_t* tileOff;      // global word offset of each (row, tile)
    uint64_t batchBase;           // global word offset of stream[0]
    int n, nTiles, row0;
    val_t* wT;                    // transposed weights: min(Kij*area_j, 1) or 0
    uint8_t* tauT;                // transposed time delays
    unsigned long long* nonZero;  // count of Kij > EPSILON
};

__global__ void __launch_bounds__(FF_BLOCK)
formFactorKernel(FormFactorParams p) {
    const int i = p.row0 + blockIdx.y;
    const int j = blockIdx.x * FF_BLOCK + threadIdx.x;
    const int n = p.n;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __shared__ uint32_t warpCnt[FF_BLOCK / 32];

    const bool valid = j < n;
    const Triangle triI = p.tris[i];
    Triangle triJ;
    if (valid) triJ = p.tris[j];

    const bool pairActive = valid && j != i && !devCulled(triI, triJ);

    // Rank of this pair among RNG-consuming pairs of the tile
    uint32_t ballot = __ballot_sync(0xffffffffu, pairActive);
    uint32_t rankInWarp = __popc(ballot & ((1u << lane) - 1u));
    if (lane == 0) warpCnt[warp] = __popc(ballot);
    __syncthreads();
    uint32_t warpBase = 0;
    for (int w = 0; w < warp; ++w) warpBase += warpCnt[w];

    val_t kij = ZERO;
    if (pairActive) {
        uint64_t off = p.tileOff[static_cast<size_t>(i) * p.nTiles + blockIdx.x] +
                       static_cast<uint64_t>(warpBase + rankInWarp) * RNG_PER_PAIR;
        const uint4* rnd = reinterpret_cast<const uint4*>(p.stream + (off - p.batchBase));

        for (int r = 0; r < NUM_RAYS; ++r) {
            uint4 w = rnd[r];
            Vec3 pI = devRandomPoint(triI, devTemperedUniform(w.x), devTemperedUniform(w.y));
            Vec3 pJ = devRandomPoint(triJ, devTemperedUniform(w.z), devTemperedUniform(w.w));

            if (devIsRayBlocked(pI, pJ, p.nodes, p.leafTris, i, j)) continue;

            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (distSqr < EPSILON) continue;

            val_t vNorm = v.norm();
            val_t cosI = vNorm <= EPSILON ? ZERO : fmaxf(ZERO, v.dot(triI.normal()) / vNorm);
            Vec3 nv = -v;
            val_t nNorm = nv.norm();
            val_t cosJ = nNorm <= EPSILON ? ZERO : fmaxf(ZERO, nv.dot(triJ.normal()) / nNorm);

            if (cosI <= ZERO || cosJ <= ZERO) continue;

            kij += (cosI * cosJ) / (PI * distSqr);
        }
        kij *= INV_NUM_RAYS;
    }

    if (valid) {
        size_t t_idx = static_cast<size_t>(j) * n + i;
        p.wT[t_idx] = kij > ZERO ? fminf(kij * p.areas[j], ONE) : ZERO;
        if (j == i) {
            p.tauT[t_idx] = TAU_SKIP;
        } else {
            val_t dist = (triI.center() - triJ.center()).norm();
            int tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
            p.tauT[t_idx] = static_cast<uint8_t>(min(tau, 254));
        }
    }

    uint32_t nzb = __ballot_sync(0xffffffffu, valid && kij > EPSILON);
    if (lane == 0 && nzb) atomicAdd(p.nonZero, static_cast<unsigned long long>(__popc(nzb)));
}

// ============================================================================
// Device Code: Wave Propagation and Cross-Correlation
// ============================================================================

// One timestep. Each thread owns one receiver triangle and accumulates over
// emitters in ascending j order (same summation order as the sequential code).
__global__ void __launch_bounds__(32)
simStepKernel(int t, int n, int sourceIdx, int timeOff, const val_t* __restrict__ wT,
              const uint8_t* __restrict__ tauT, const val_t* rho, val_t* radB) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t sumB = ZERO;
    #pragma unroll 16
    for (int j = 0; j < n; ++j) {
        size_t k = static_cast<size_t>(j) * n + i;
        int tau = tauT[k];
        if (t < tau) continue;
        val_t w = wT[k];
        val_t radJ = radB[static_cast<size_t>(t - tau) * n + j];
        if (radJ <= ZERO) continue;
        sumB += w * radJ;
    }
    val_t emit = (i == sourceIdx && t < timeOff) ? 1.0f : ZERO;
    radB[static_cast<size_t>(t) * n + i] = rho[i] * sumB + emit;
}

// One block per receiver triangle; threads cover lags t.
__global__ void distanceKernel(int n, int steps, int sourceIdx, const val_t* radB, val_t* distances) {
    int i = blockIdx.x;
    val_t best = ZERO;
    int bestT = 0;
    for (int t = threadIdx.x; t < steps; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < steps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * n + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * n + sourceIdx];
            sum += pS * pB;
        }
        if (sum > best) {
            best = sum;
            bestT = t;
        }
    }
    __shared__ val_t sBest[256];
    __shared__ int sT[256];
    sBest[threadIdx.x] = best;
    sT[threadIdx.x] = bestT;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            val_t ob = sBest[threadIdx.x + s];
            int ot = sT[threadIdx.x + s];
            if (ob > sBest[threadIdx.x] || (ob == sBest[threadIdx.x] && ot < sT[threadIdx.x])) {
                sBest[threadIdx.x] = ob;
                sT[threadIdx.x] = ot;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[i] = WAVE_SPEED * static_cast<val_t>(sT[0]);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), copied back on demand
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure (host build)
    DevOctree devOctree;            // Flattened copy for the GPU

    size_t sourceIndex;
    size_t nonZeroKij = 0;          // Number of form factors > EPSILON

    // Device buffers
    Triangle* dTris = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    DevNode* dNodes = nullptr;
    LeafTri* dLeafTris = nullptr;
    val_t* dWT = nullptr;
    uint8_t* dTauT = nullptr;
    val_t* dRadB = nullptr;
    val_t* dDistances = nullptr;

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
    flattenOctree(state.octree, state.triangles, state.devOctree);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);
}

void uploadToDevice(SimulationState& state) {
    const size_t n = state.numTriangles;
    CUDA_CHECK(cudaMalloc(&state.dTris, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRho, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, state.devOctree.nodes.size() * sizeof(DevNode)));
    CUDA_CHECK(cudaMalloc(&state.dLeafTris, std::max<size_t>(1, state.devOctree.tris.size()) * sizeof(LeafTri)));
    CUDA_CHECK(cudaMalloc(&state.dWT, n * n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dTauT, n * n * sizeof(uint8_t)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, std::max<size_t>(1, state.numTimesteps * n) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dDistances, n * sizeof(val_t)));

    CUDA_CHECK(cudaMemcpy(state.dTris, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, state.devOctree.nodes.data(),
                          state.devOctree.nodes.size() * sizeof(DevNode), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dLeafTris, state.devOctree.tris.data(),
                          state.devOctree.tris.size() * sizeof(LeafTri), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, state.numTimesteps * n * sizeof(val_t)));
}

void freeDevice(SimulationState& state) {
    cudaFree(state.dTris); cudaFree(state.dAreas); cudaFree(state.dRho);
    cudaFree(state.dNodes); cudaFree(state.dLeafTris); cudaFree(state.dWT);
    cudaFree(state.dTauT); cudaFree(state.dRadB); cudaFree(state.dDistances);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

// Computes form factors (as weights) and time delays on the GPU. The random
// stream of the sequential algorithm (one mt19937 consumed pair by pair) is
// reproduced exactly: each non-culled pair owns a fixed 64-word slice of it.
void computeFormFactors(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    printf("Computing form factors (Kij)...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int nTiles = (n + FF_BLOCK - 1) / FF_BLOCK;
    const size_t numTiles = static_cast<size_t>(n) * nTiles;

    // Pass 1: count RNG-consuming pairs per (row, tile) and scan on host
    uint32_t* dCnt;
    CUDA_CHECK(cudaMalloc(&dCnt, numTiles * sizeof(uint32_t)));
    countPairsKernel<<<dim3(nTiles, n), FF_BLOCK>>>(state.dTris, n, nTiles, dCnt);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> cnt(numTiles);
    CUDA_CHECK(cudaMemcpy(cnt.data(), dCnt, numTiles * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dCnt));

    std::vector<uint64_t> tileOff(numTiles);
    std::vector<uint64_t> rowStart(n + 1);
    uint64_t acc = 0;
    for (int i = 0; i < n; ++i) {
        rowStart[i] = acc;
        for (int tl = 0; tl < nTiles; ++tl) {
            size_t k = static_cast<size_t>(i) * nTiles + tl;
            tileOff[k] = acc;
            acc += static_cast<uint64_t>(cnt[k]) * RNG_PER_PAIR;
        }
    }
    rowStart[n] = acc;

    uint64_t* dTileOff;
    unsigned long long* dNonZero;
    CUDA_CHECK(cudaMalloc(&dTileOff, numTiles * sizeof(uint64_t)));
    CUDA_CHECK(cudaMemcpy(dTileOff, tileOff.data(), numTiles * sizeof(uint64_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMalloc(&dNonZero, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(dNonZero, 0, sizeof(unsigned long long)));

    // Pass 2: batches of rows, host generates the MT stream while the GPU consumes
    const uint64_t maxRow = static_cast<uint64_t>(n) * RNG_PER_PAIR;
    const uint64_t cap = std::max<uint64_t>(maxRow, std::min<uint64_t>(1ull << 25, std::max<uint64_t>(acc, 1)));
    const uint64_t bufWords = std::min<uint64_t>(cap, std::max<uint64_t>(acc, 4));

    uint32_t* hBuf[2];
    uint32_t* dBuf[2];
    cudaStream_t streams[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&hBuf[b], bufWords * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&dBuf[b], bufWords * sizeof(uint32_t)));
        CUDA_CHECK(cudaStreamCreate(&streams[b]));
    }

    struct Batch { int r0, r1; };
    std::vector<Batch> batches;
    for (int r0 = 0; r0 < n;) {
        int r1 = r0 + 1;
        while (r1 < n && rowStart[r1 + 1] - rowStart[r0] <= bufWords) ++r1;
        batches.push_back({r0, r1});
        r0 = r1;
    }

    auto report = [&](const Batch& bt) {
        for (int i = bt.r0; i < bt.r1; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) printf("  Progress: %d/%d triangles\n", i + 1, n);
        }
    };

    MtRawStream gen(42);
    const size_t nb = batches.size();
    for (size_t k = 0; k < nb; ++k) {
        const int b = static_cast<int>(k & 1);
        if (k >= 2) {
            CUDA_CHECK(cudaStreamSynchronize(streams[b]));  // buffer b is free again
            report(batches[k - 2]);
        }
        const Batch& bt = batches[k];
        uint64_t base = rowStart[bt.r0];
        uint64_t words = rowStart[bt.r1] - base;

        gen.fill(hBuf[b], words);
        if (words > 0) {
            CUDA_CHECK(cudaMemcpyAsync(dBuf[b], hBuf[b], words * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, streams[b]));
        }

        FormFactorParams p;
        p.tris = state.dTris; p.areas = state.dAreas;
        p.nodes = state.dNodes; p.leafTris = state.dLeafTris;
        p.stream = dBuf[b]; p.tileOff = dTileOff; p.batchBase = base;
        p.n = n; p.nTiles = nTiles; p.row0 = bt.r0;
        p.wT = state.dWT; p.tauT = state.dTauT; p.nonZero = dNonZero;
        formFactorKernel<<<dim3(nTiles, bt.r1 - bt.r0), FF_BLOCK, 0, streams[b]>>>(p);
        CUDA_CHECK(cudaGetLastError());
    }
    for (size_t k = (nb >= 2 ? nb - 2 : 0); k < nb; ++k) {
        CUDA_CHECK(cudaStreamSynchronize(streams[k & 1]));
        report(batches[k]);
    }

    unsigned long long nz = 0;
    CUDA_CHECK(cudaMemcpy(&nz, dNonZero, sizeof(nz), cudaMemcpyDeviceToHost));
    state.nonZeroKij = static_cast<size_t>(nz);

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaStreamDestroy(streams[b]));
        CUDA_CHECK(cudaFreeHost(hBuf[b]));
        CUDA_CHECK(cudaFree(dBuf[b]));
    }
    CUDA_CHECK(cudaFree(dTileOff));
    CUDA_CHECK(cudaFree(dNonZero));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int steps = static_cast<int>(state.numTimesteps);
    const int timeOff = steps / 2;
    const int blocks = (n + 31) / 32;

    for (int t = 0; t < steps; ++t) {
        simStepKernel<<<blocks, 32>>>(t, n, static_cast<int>(state.sourceIndex), timeOff,
                                      state.dWT, state.dTauT, state.dRho, state.dRadB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == steps) {
            CUDA_CHECK(cudaDeviceSynchronize());
            printf("  Timestep %d/%d\n", t + 1, steps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const int n = static_cast<int>(state.numTriangles);
    distanceKernel<<<n, 256>>>(n, static_cast<int>(state.numTimesteps),
                               static_cast<int>(state.sourceIndex), state.dRadB, state.dDistances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances, n * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, state.radB.size() * sizeof(val_t),
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
    size_t nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %zu/%zu (%.2f%%)\n",
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

    // Initialize
    CUDA_CHECK(cudaSetDevice(0));
    CUDA_CHECK(cudaFree(0));  // Create context up front

    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    uploadToDevice(state);
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
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
