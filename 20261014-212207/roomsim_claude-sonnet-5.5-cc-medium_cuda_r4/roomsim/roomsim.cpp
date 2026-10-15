/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified implementation (parallelized with CUDA) of room impulse response
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
#include <cfloat>
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

#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err_ = (call);                                             \
        if (err_ != cudaSuccess) {                                             \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                      \
                    cudaGetErrorString(err_), __FILE__, __LINE__);             \
            exit(1);                                                           \
        }                                                                      \
    } while (0)

#define HD __host__ __device__

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
// Mersenne Twister stream (bit-identical to std::mt19937 + uniform_real<float>)
// ============================================================================

// The original code draws every random number from one sequential mt19937
// stream. To keep identical semantics, the raw 32-bit outputs are generated in
// order on the host and converted to floats on the GPU, where each triangle
// pair knows its exact offset into the stream.
class FastMT {
    static constexpr int N = 624, M = 397;
    uint32_t mt[N];
    int pos = N;

    static inline uint32_t mix(uint32_t u, uint32_t l) {
        uint32_t y = (u & 0x80000000u) | (l & 0x7fffffffu);
        return (y >> 1) ^ ((0u - (y & 1u)) & 0x9908b0dfu);
    }
    void twist() {
        for (int k = 0; k < N - M; ++k) mt[k] = mt[k + M] ^ mix(mt[k], mt[k + 1]);
        for (int k = N - M; k < N - 1; ++k) mt[k] = mt[k - (N - M)] ^ mix(mt[k], mt[k + 1]);
        mt[N - 1] = mt[M - 1] ^ mix(mt[N - 1], mt[0]);
        pos = 0;
    }

public:
    explicit FastMT(uint32_t seed = 42) {
        mt[0] = seed;
        for (int i = 1; i < N; ++i) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
    }
    void fill(uint32_t* out, size_t n) {
        while (n > 0) {
            if (pos == N) twist();
            size_t take = std::min(n, static_cast<size_t>(N - pos));
            const uint32_t* src = mt + pos;
            for (size_t k = 0; k < take; ++k) {
                uint32_t y = src[k];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680u;
                y ^= (y << 15) & 0xefc60000u;
                y ^= y >> 18;
                out[k] = y;
            }
            out += take;
            pos += static_cast<int>(take);
            n -= take;
        }
    }
};

// std::generate_canonical<float, 24> on a 32-bit engine
__device__ __forceinline__ val_t toUnitFloat(uint32_t x) {
    val_t r = __uint2float_rn(x) * 2.3283064365386963e-10f;  // * 2^-32 (exact)
    return r >= 1.0f ? 0.99999994f : r;
}

// ============================================================================
// Device-side octree, geometry and form factor kernels
// ============================================================================

struct GpuNode {
    Vec3 center, halfExtent;
    int child[8];
    int leafStart, leafCount;
};

constexpr int GPU_STACK = 64;
constexpr int RAYS_PER_PAIR = NUM_RAYS;           // one lane per ray
constexpr int DRAWS_PER_RAY = 4;
constexpr int DRAWS_PER_PAIR = NUM_RAYS * DRAWS_PER_RAY;
constexpr int KIJ_BLOCK = 128;
constexpr int KIJ_PAIRS_PER_BLOCK = KIJ_BLOCK / RAYS_PER_PAIR;

// Flatten the pointer-based octree for upload. Returns node index.
int flattenOctree(const Octree& o, std::vector<GpuNode>& nodes, std::vector<int>& leafIdx,
                  int depth, int& maxDepth) {
    maxDepth = std::max(maxDepth, depth);
    int me = static_cast<int>(nodes.size());
    nodes.emplace_back();
    GpuNode nd{};
    nd.center = o.center;
    nd.halfExtent = o.halfExtent;
    nd.leafStart = 0;
    nd.leafCount = 0;
    for (int i = 0; i < 8; ++i) nd.child[i] = -1;
    if (!o.triangleIndices.empty()) {
        nd.leafStart = static_cast<int>(leafIdx.size());
        nd.leafCount = static_cast<int>(o.triangleIndices.size());
        for (size_t idx : o.triangleIndices) leafIdx.push_back(static_cast<int>(idx));
    } else {
        for (int i = 0; i < 8; ++i) {
            if (o.children[i]) nd.child[i] = flattenOctree(*o.children[i], nodes, leafIdx, depth + 1, maxDepth);
        }
    }
    nodes[me] = nd;
    return me;
}

__device__ __forceinline__ bool nodeHitsRay(const GpuNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - n.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    const Vec3& h = n.halfExtent;

    if (fabsf(c.x) > h.x + ad.x) return false;
    if (fabsf(c.y) > h.y + ad.y) return false;
    if (fabsf(c.z) > h.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > h.y * ad.z + h.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > h.z * ad.x + h.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > h.x * ad.y + h.y * ad.x + EPSILON) return false;
    return true;
}

// Moller-Trumbore
__device__ __forceinline__ val_t rayTriangleIntersectDev(const Vec3& orig, const Vec3& dir,
                                                          const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return e2.dot(qvec) * invDet;
}

__device__ bool isRayBlockedDev(const Vec3& from, const Vec3& to, const GpuNode* nodes,
                                const int* leafIdx, const Triangle* tris, int src, int dst) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[GPU_STACK];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const GpuNode& nd = nodes[stack[--sp]];
        if (nd.leafCount > 0) {
            for (int k = 0; k < nd.leafCount; ++k) {
                int idx = leafIdx[nd.leafStart + k];
                if (idx == src || idx == dst) continue;
                const Triangle& t = tris[idx];
                val_t dist = rayTriangleIntersectDev(from, dirNorm, t.a, t.b, t.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 7; c >= 0; --c) {
                int ci = nd.child[c];
                if (ci >= 0 && nodeHitsRay(nodes[ci], from, to)) stack[sp++] = ci;
            }
        }
    }
    return false;
}

__device__ __forceinline__ val_t cosPhiDev(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

__device__ __forceinline__ Vec3 randomPointInTriangleDev(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// For every row i: list the columns j that are not culled (same facing / diagonal).
// If cols == nullptr only the per-row counts are produced.
__global__ void compactRowsKernel(const Triangle* __restrict__ tris, int n, int rowStart,
                                  uint32_t* __restrict__ cols, uint32_t* __restrict__ rowCnt) {
    __shared__ uint32_t warpCnt[32];
    __shared__ uint32_t base;
    const int i = rowStart + blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31, warp = tid >> 5, nWarps = blockDim.x >> 5;
    const Vec3 ni = tris[i]._normal;
    if (tid == 0) base = 0;
    __syncthreads();
    for (int j0 = 0; j0 < n; j0 += blockDim.x) {
        int j = j0 + tid;
        bool flag = j < n && j != i && !(ni.dot(tris[j]._normal) > 0.99f);
        uint32_t ballot = __ballot_sync(0xffffffffu, flag);
        if (lane == 0) warpCnt[warp] = __popc(ballot);
        __syncthreads();
        uint32_t off = 0, total = 0;
        for (int w = 0; w < nWarps; ++w) {
            if (w < warp) off += warpCnt[w];
            total += warpCnt[w];
        }
        if (flag && cols) {
            cols[static_cast<size_t>(blockIdx.x) * n + base + off + __popc(ballot & ((1u << lane) - 1u))] = j;
        }
        __syncthreads();
        if (tid == 0) base += total;
        __syncthreads();
    }
    if (tid == 0 && !cols) rowCnt[i] = base;
}

// One lane per ray, 16 lanes per triangle pair. Output is stored transposed
// (kijT[j * n + i]) so the simulation kernel reads coalesced.
__global__ void __launch_bounds__(KIJ_BLOCK)
formFactorKernel(const Triangle* __restrict__ tris, const GpuNode* __restrict__ nodes,
                 const int* __restrict__ leafIdx, const uint32_t* __restrict__ cols,
                 const uint64_t* __restrict__ rowDrawBase, const uint32_t* __restrict__ rowCnt,
                 const uint32_t* __restrict__ draws, uint64_t chunkBase, int rowStart, int n,
                 val_t* __restrict__ kijT) {
    const int i = rowStart + blockIdx.x;
    const int cnt = static_cast<int>(rowCnt[i]);
    const int firstPair = blockIdx.y * KIJ_PAIRS_PER_BLOCK;
    if (firstPair >= cnt) return;

    const int ray = threadIdx.x & (RAYS_PER_PAIR - 1);
    const int k = firstPair + (threadIdx.x / RAYS_PER_PAIR);
    const bool active = k < cnt;
    const int kc = active ? k : cnt - 1;
    const int j = static_cast<int>(cols[static_cast<size_t>(blockIdx.x) * n + kc]);

    val_t term = ZERO;
    if (active) {
        const Triangle& triI = tris[i];
        const Triangle& triJ = tris[j];
        const uint64_t off = rowDrawBase[i] - chunkBase + static_cast<uint64_t>(k) * DRAWS_PER_PAIR +
                             static_cast<uint64_t>(ray) * DRAWS_PER_RAY;
        uint4 d = *reinterpret_cast<const uint4*>(draws + off);
        Vec3 pI = randomPointInTriangleDev(triI, toUnitFloat(d.x), toUnitFloat(d.y));
        Vec3 pJ = randomPointInTriangleDev(triJ, toUnitFloat(d.z), toUnitFloat(d.w));

        if (!isRayBlockedDev(pI, pJ, nodes, leafIdx, tris, i, j)) {
            Vec3 v = pJ - pI;
            val_t distSqr = v.squaredNorm();
            if (!(distSqr < EPSILON)) {
                val_t cosPhiI = cosPhiDev(v, triI._normal);
                val_t cosPhiJ = cosPhiDev(-v, triJ._normal);
                if (!(cosPhiI <= ZERO || cosPhiJ <= ZERO)) {
                    term = (cosPhiI * cosPhiJ) / (PI * distSqr);
                }
            }
        }
    }

    // Ordered reduction across the 16 rays (same summation order as sequential code)
    val_t sum = ZERO;
#pragma unroll
    for (int r = 0; r < RAYS_PER_PAIR; ++r) {
        sum += __shfl_sync(0xffffffffu, term, (threadIdx.x & 16) + r);
    }
    if (active && ray == 0) kijT[static_cast<size_t>(j) * n + i] = sum * INV_NUM_RAYS;
}

// Time delays, stored transposed: tauT[j * n + i]
__global__ void tauKernel(const Vec3* __restrict__ centers, int n, int* __restrict__ tauT) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y;
    if (i >= n || i == j) return;
    val_t dist = (centers[i] - centers[j]).norm();
    tauT[static_cast<size_t>(j) * n + i] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// One wave-propagation timestep: one thread per receiving triangle i, summing
// over j in ascending order (identical accumulation order to the sequential code).
constexpr int SIM_UNROLL = 8;
__global__ void simStepKernel(const val_t* __restrict__ kijT, const int* __restrict__ tauT,
                              const val_t* __restrict__ areas, const val_t* __restrict__ rho,
                              val_t* radB, int n, int t, int sourceIdx, int timeOff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t sumB = ZERO;
    int j = 0;
    for (; j + SIM_UNROLL <= n; j += SIM_UNROLL) {
        int tau[SIM_UNROLL];
        val_t kij[SIM_UNROLL], rad[SIM_UNROLL];
#pragma unroll
        for (int u = 0; u < SIM_UNROLL; ++u) {
            size_t o = static_cast<size_t>(j + u) * n + i;
            tau[u] = tauT[o];
            kij[u] = kijT[o];
        }
#pragma unroll
        for (int u = 0; u < SIM_UNROLL; ++u) {
            bool ok = t >= tau[u] && kij[u] > ZERO;
            rad[u] = ok ? radB[static_cast<size_t>(t - tau[u]) * n + j + u] : ZERO;
        }
#pragma unroll
        for (int u = 0; u < SIM_UNROLL; ++u) {
            if (rad[u] > ZERO) {
                val_t w = kij[u] * areas[j + u];
                sumB += (ONE < w ? ONE : w) * rad[u];
            }
        }
    }
    for (; j < n; ++j) {
        size_t o = static_cast<size_t>(j) * n + i;
        int tau = tauT[o];
        val_t kij = kijT[o];
        if (t < tau || kij <= ZERO) continue;
        val_t radJ = radB[static_cast<size_t>(t - tau) * n + j];
        if (radJ <= ZERO) continue;
        val_t w = kij * areas[j];
        sumB += (ONE < w ? ONE : w) * radJ;
    }
    val_t radE = (i == sourceIdx && t < timeOff) ? 1.0f : ZERO;
    radB[static_cast<size_t>(t) * n + i] = rho[i] * sumB + radE;
}

// Cross-correlation: one block per triangle, threads over lags.
constexpr int DIST_BLOCK = 128;
__global__ void distanceKernel(const val_t* __restrict__ radB, const val_t* __restrict__ srcSig,
                               int n, int timesteps, val_t* __restrict__ distances) {
    __shared__ val_t sVal[DIST_BLOCK];
    __shared__ int sT[DIST_BLOCK];
    const int i = blockIdx.x;
    val_t best = ZERO;
    int bestT = 0;
    for (int t = threadIdx.x; t < timesteps; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < timesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * n + i];
            val_t pS = srcSig[tt - t];
            sum += pS * pB;
        }
        if (sum > best) {
            best = sum;
            bestT = t;
        }
    }
    sVal[threadIdx.x] = best;
    sT[threadIdx.x] = bestT;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            val_t ov = sVal[threadIdx.x + s];
            int ot = sT[threadIdx.x + s];
            if (ov > sVal[threadIdx.x] || (ov == sVal[threadIdx.x] && ot < sT[threadIdx.x])) {
                sVal[threadIdx.x] = ov;
                sT[threadIdx.x] = ot;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[i] = WAVE_SPEED * static_cast<val_t>(sT[0]);
}

__global__ void countNonZeroKernel(const val_t* __restrict__ kij, size_t total, unsigned long long* count) {
    unsigned long long local = 0;
    for (size_t i = blockIdx.x * static_cast<size_t>(blockDim.x) + threadIdx.x; i < total;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (kij[i] > EPSILON) local++;
    }
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(count, local);
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), only kept for validation
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;
    size_t timeOff;                 // Source emission active for t < timeOff
    bool keepResults = false;       // Copy intermediate data back for validation
    unsigned long long nonZeroKij = 0;

    // Device data
    Triangle* dTris = nullptr;
    Vec3* dCenters = nullptr;
    GpuNode* dNodes = nullptr;
    int* dLeafIdx = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;
    val_t* dKijT = nullptr;         // Form factors (N x N, transposed: [j * N + i])
    int* dTauT = nullptr;           // Time delays (N x N, transposed: [j * N + i])
    val_t* dRadB = nullptr;         // T x N
    val_t* dDistances = nullptr;
};

template <typename T>
T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, std::max<size_t>(count, 1) * sizeof(T)));
    return p;
}

template <typename T>
T* deviceUpload(const std::vector<T>& v) {
    T* p = deviceAlloc<T>(v.size());
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

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

    // Initialize areas
    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);
    state.distances.resize(state.numTriangles, ZERO);

    // Source emission (active for first half of timesteps)
    state.timeOff = timesteps / 2;

    // Upload static data to the GPU
    std::vector<Vec3> centers(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) centers[i] = state.triangles[i].center();

    std::vector<GpuNode> nodes;
    std::vector<int> leafIdx;
    int maxDepth = 0;
    flattenOctree(state.octree, nodes, leafIdx, 0, maxDepth);
    if (7 * maxDepth + 1 > GPU_STACK) {
        fprintf(stderr, "Octree too deep for device traversal stack (%d)\n", maxDepth);
        exit(1);
    }

    state.dTris = deviceUpload(state.triangles);
    state.dCenters = deviceUpload(centers);
    state.dNodes = deviceUpload(nodes);
    state.dLeafIdx = deviceUpload(leafIdx);
    state.dAreas = deviceUpload(state.areas);
    state.dRho = deviceUpload(state.rho);
    state.dDistances = deviceAlloc<val_t>(state.numTriangles);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int n = static_cast<int>(state.numTriangles);
    const size_t nn = static_cast<size_t>(n) * n;

    state.dKijT = deviceAlloc<val_t>(nn);
    CUDA_CHECK(cudaMemset(state.dKijT, 0, nn * sizeof(val_t)));

    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

    // Number of non-culled pairs per row determines how many random numbers
    // the sequential algorithm would have consumed before each row.
    uint32_t* dRowCnt = deviceAlloc<uint32_t>(n);
    compactRowsKernel<<<n, 256, 0, stream>>>(state.dTris, n, 0, nullptr, dRowCnt);
    CUDA_CHECK(cudaGetLastError());
    std::vector<uint32_t> rowCnt(n);
    CUDA_CHECK(cudaMemcpyAsync(rowCnt.data(), dRowCnt, n * sizeof(uint32_t), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    std::vector<uint64_t> rowDrawBase(n + 1, 0);
    for (int i = 0; i < n; ++i) {
        rowDrawBase[i + 1] = rowDrawBase[i] + static_cast<uint64_t>(rowCnt[i]) * DRAWS_PER_PAIR;
    }
    uint64_t* dRowDrawBase = deviceUpload(rowDrawBase);

    // Plan chunks of rows, bounded by random-number buffer and column list size
    const uint64_t maxChunkDraws = 32ull << 20;
    const uint64_t maxChunkCols = 32ull << 20;
    struct Chunk { int rowStart, rowEnd; };
    std::vector<Chunk> chunks;
    uint64_t bufDraws = 0;
    size_t maxRows = 0;
    for (int r = 0; r < n;) {
        int e = r + 1;
        while (e < n && rowDrawBase[e + 1] - rowDrawBase[r] <= maxChunkDraws &&
               static_cast<uint64_t>(e + 1 - r) * n <= maxChunkCols) {
            ++e;
        }
        chunks.push_back({r, e});
        bufDraws = std::max(bufDraws, rowDrawBase[e] - rowDrawBase[r]);
        maxRows = std::max<size_t>(maxRows, e - r);
        r = e;
    }

    uint32_t* pinned[2];
    uint32_t* dDraws[2];
    cudaEvent_t done[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&pinned[b], std::max<uint64_t>(bufDraws, 4) * sizeof(uint32_t)));
        dDraws[b] = deviceAlloc<uint32_t>(std::max<uint64_t>(bufDraws, 4));
        CUDA_CHECK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
    }
    uint32_t* dCols = deviceAlloc<uint32_t>(maxRows * static_cast<size_t>(n));

    FastMT mt(42);
    for (size_t c = 0; c < chunks.size(); ++c) {
        const int b = static_cast<int>(c & 1);
        const int r0 = chunks[c].rowStart, r1 = chunks[c].rowEnd;
        const uint64_t base = rowDrawBase[r0];
        const uint64_t count = rowDrawBase[r1] - base;

        // Wait until the buffer pair from two chunks ago is free, then produce
        // this chunk's slice of the random stream on the host while the GPU
        // works on the previous chunk.
        CUDA_CHECK(cudaEventSynchronize(done[b]));
        mt.fill(pinned[b], count);
        CUDA_CHECK(cudaMemcpyAsync(dDraws[b], pinned[b], count * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, stream));

        compactRowsKernel<<<r1 - r0, 256, 0, stream>>>(state.dTris, n, r0, dCols, dRowCnt);
        uint32_t maxCnt = 0;
        for (int i = r0; i < r1; ++i) maxCnt = std::max(maxCnt, rowCnt[i]);
        if (maxCnt > 0) {
            dim3 grid(r1 - r0, (maxCnt + KIJ_PAIRS_PER_BLOCK - 1) / KIJ_PAIRS_PER_BLOCK);
            formFactorKernel<<<grid, KIJ_BLOCK, 0, stream>>>(
                state.dTris, state.dNodes, state.dLeafIdx, dCols, dRowDrawBase, dRowCnt,
                dDraws[b], base, r0, n, state.dKijT);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(done[b], stream));

        for (int i = r0; i < r1; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %d/%d triangles\n", i + 1, n);
            }
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFreeHost(pinned[b]));
        CUDA_CHECK(cudaFree(dDraws[b]));
        CUDA_CHECK(cudaEventDestroy(done[b]));
    }
    CUDA_CHECK(cudaFree(dCols));
    CUDA_CHECK(cudaFree(dRowCnt));
    CUDA_CHECK(cudaFree(dRowDrawBase));
    CUDA_CHECK(cudaStreamDestroy(stream));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int n = static_cast<int>(state.numTriangles);
    const size_t nn = static_cast<size_t>(n) * n;

    state.dTauT = deviceAlloc<int>(nn);
    CUDA_CHECK(cudaMemset(state.dTauT, 0, nn * sizeof(int)));
    dim3 grid((n + 127) / 128, n);
    tauKernel<<<grid, 128>>>(state.dCenters, n, state.dTauT);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    state.dRadB = deviceAlloc<val_t>(static_cast<size_t>(T) * n);
    CUDA_CHECK(cudaMemset(state.dRadB, 0, static_cast<size_t>(T) * n * sizeof(val_t)));

    const int block = 64;
    const int grid = (n + block - 1) / block;
    for (int t = 0; t < T; ++t) {
        simStepKernel<<<grid, block>>>(state.dKijT, state.dTauT, state.dAreas, state.dRho,
                                       state.dRadB, n, t, static_cast<int>(state.sourceIndex),
                                       static_cast<int>(state.timeOff));
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
    const int n = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    if (T > 0) {
        val_t* dSrc = deviceAlloc<val_t>(T);
        CUDA_CHECK(cudaMemcpy2D(dSrc, sizeof(val_t), state.dRadB + state.sourceIndex, n * sizeof(val_t),
                                sizeof(val_t), T, cudaMemcpyDeviceToDevice));
        distanceKernel<<<n, DIST_BLOCK>>>(state.dRadB, dSrc, n, T, state.dDistances);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaFree(dSrc));
    } else {
        CUDA_CHECK(cudaMemset(state.dDistances, 0, n * sizeof(val_t)));
    }
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDistances, n * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
}

// Bring back what validation needs (outside of the timed regions)
void fetchValidationData(SimulationState& state) {
    const size_t n = state.numTriangles;
    state.radB.resize(state.numTimesteps * n);
    if (!state.radB.empty()) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, state.radB.size() * sizeof(val_t),
                              cudaMemcpyDeviceToHost));
    }
    unsigned long long* dCount = deviceAlloc<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(dCount, 0, sizeof(unsigned long long)));
    countNonZeroKernel<<<1024, 256>>>(state.dKijT, n * n, dCount);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&state.nonZeroKij, dCount, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dCount));
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
            if (state.radB[t * state.numTriangles + i] > EPSILON) {
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
    unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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
    CUDA_CHECK(cudaFree(0));  // create the CUDA context before timing

    SimulationState state;
    state.keepResults = validate;
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

    if (validate) fetchValidationData(state);

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
