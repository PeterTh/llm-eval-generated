/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error '%s' at %s:%d\n",                         \
                    cudaGetErrorString(err_), __FILE__, __LINE__);                \
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
    HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

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

    HD Triangle() = default;
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
// Host-side MT19937 (state advance only; used to place checkpoints so the GPU
// reproduces the exact std::mt19937 stream of the sequential program)
// ============================================================================

constexpr int MT_N = 624;
constexpr int MT_M = 397;
constexpr int MT_SEG_TWISTS = 16;  // twist blocks generated per GPU segment

static void mtSeed(uint32_t* mt, uint32_t seed) {
    mt[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
    }
}

static inline uint32_t mtMix(uint32_t u, uint32_t v) {
    return (((u & 0x80000000u) | (v & 0x7fffffffu)) >> 1) ^ ((0u - (v & 1u)) & 0x9908b0dfu);
}

static void mtTwist(uint32_t* __restrict mt) {
    for (int k = 0; k < MT_N - MT_M; ++k) {
        mt[k] = mt[k + MT_M] ^ mtMix(mt[k], mt[k + 1]);
    }
    #pragma GCC ivdep
    for (int k = MT_N - MT_M; k < MT_N - 1; ++k) {
        mt[k] = mt[k - (MT_N - MT_M)] ^ mtMix(mt[k], mt[k + 1]);
    }
    mt[MT_N - 1] = mt[MT_M - 1] ^ mtMix(mt[MT_N - 1], mt[0]);
}

// ============================================================================
// Octree flattening for the device
// ============================================================================

struct DevNode {
    float cx, cy, cz;
    float hx, hy, hz;
    int leafStart, leafCount;
    int child[8];
};

struct LeafTri {
    float ax, ay, az;
    float e1x, e1y, e1z;
    float e2x, e2y, e2z;
    int idx;
};

constexpr int TRAVERSAL_STACK = 64;

static int flattenOctree(const Octree& node, std::vector<DevNode>& nodes,
                         std::vector<LeafTri>& leaves, int depth, int& maxDepth) {
    maxDepth = std::max(maxDepth, depth);
    int me = static_cast<int>(nodes.size());
    nodes.emplace_back();
    DevNode dn{};
    dn.cx = node.center.x; dn.cy = node.center.y; dn.cz = node.center.z;
    dn.hx = node.halfExtent.x; dn.hy = node.halfExtent.y; dn.hz = node.halfExtent.z;
    dn.leafStart = static_cast<int>(leaves.size());
    dn.leafCount = static_cast<int>(node.triangleIndices.size());
    for (int i = 0; i < 8; ++i) dn.child[i] = -1;

    for (size_t idx : node.triangleIndices) {
        const Triangle& t = (*node.allTriangles)[idx];
        Vec3 e1 = t.b - t.a;
        Vec3 e2 = t.c - t.a;
        leaves.push_back({t.a.x, t.a.y, t.a.z, e1.x, e1.y, e1.z, e2.x, e2.y, e2.z,
                          static_cast<int>(idx)});
    }
    if (dn.leafCount == 0) {
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                dn.child[i] = flattenOctree(*node.children[i], nodes, leaves, depth + 1, maxDepth);
            }
        }
    }
    nodes[me] = dn;
    return me;
}

// ============================================================================
// Device code
// ============================================================================

struct Entry {
    float w;   // min(Kij * area_j, 1), stored transposed: mat[j * N + i]
    int tau;   // time delay, same layout
};

__device__ __forceinline__ float rayTriangleIntersectD(const Vec3& orig, const Vec3& dir,
                                                       const LeafTri& t) {
    Vec3 v0(t.ax, t.ay, t.az);
    Vec3 e1(t.e1x, t.e1y, t.e1z);
    Vec3 e2(t.e2x, t.e2y, t.e2z);
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

__device__ __forceinline__ bool rayIntersectsBoxD(const DevNode& n, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - Vec3(n.cx, n.cy, n.cz);
    Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > n.hx + ad.x) return false;
    if (fabsf(c.y) > n.hy + ad.y) return false;
    if (fabsf(c.z) > n.hz + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > n.hy * ad.z + n.hz * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > n.hz * ad.x + n.hx * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > n.hx * ad.y + n.hy * ad.x + EPSILON) return false;

    return true;
}

__device__ bool isRayBlockedD(const Vec3& from, const Vec3& to,
                              const DevNode* __restrict__ nodes,
                              const LeafTri* __restrict__ leaves,
                              int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[TRAVERSAL_STACK];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const DevNode& n = nodes[stack[--sp]];
        if (n.leafCount > 0) {
            for (int k = 0; k < n.leafCount; ++k) {
                const LeafTri& t = leaves[n.leafStart + k];
                if (t.idx == srcTriIdx || t.idx == dstTriIdx) continue;
                val_t dist = rayTriangleIntersectD(from, dirNorm, t);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int i = 7; i >= 0; --i) {
                int c = n.child[i];
                if (c >= 0 && rayIntersectsBoxD(nodes[c], from, to)) stack[sp++] = c;
            }
        }
    }
    return false;
}

__device__ __forceinline__ Vec3 randomPointInTriangleD(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__device__ __forceinline__ val_t cosPhiD(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Counts, for every row i, the number of pairs (i, j) that consume random numbers
__global__ void countPairsKernel(const Triangle* __restrict__ tris, int n,
                                 unsigned* __restrict__ counts) {
    int i = blockIdx.x;
    Vec3 nI = tris[i].normal();
    unsigned c = 0;
    for (int j = threadIdx.x; j < n; j += blockDim.x) {
        if (j != i && !(nI.dot(tris[j].normal()) > 0.99f)) ++c;
    }
    for (int o = 16; o > 0; o >>= 1) c += __shfl_down_sync(0xffffffffu, c, o);
    __shared__ unsigned part[32];
    if ((threadIdx.x & 31) == 0) part[threadIdx.x >> 5] = c;
    __syncthreads();
    if (threadIdx.x == 0) {
        unsigned total = 0;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) total += part[w];
        counts[i] = total;
    }
}

// Time delays for all pairs, and zero-initialised form factors
__global__ void initMatrixKernel(const Triangle* __restrict__ tris, int n, Entry* __restrict__ mat) {
    int j = blockIdx.x;
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= n) return;
    Entry e;
    e.w = ZERO;
    e.tau = 0;
    if (i != j) {
        val_t dist = (tris[i].center() - tris[j].center()).norm();
        e.tau = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    mat[static_cast<size_t>(j) * n + i] = e;
}

// Generates MT19937 outputs (as floats in [0,1), identical to
// std::uniform_real_distribution<float>) for MT_SEG_TWISTS twist blocks per CUDA block,
// starting from the checkpointed state at the segment start.
__global__ void __launch_bounds__(256) mtGenKernel(const uint32_t* __restrict__ cp,
                                                   float* __restrict__ out) {
    __shared__ uint32_t mt[MT_N];
    const int tid = threadIdx.x;
    const uint32_t* src = cp + static_cast<size_t>(blockIdx.x) * MT_N;
    float* dst = out + static_cast<size_t>(blockIdx.x) * MT_SEG_TWISTS * MT_N;

    for (int k = tid; k < MT_N; k += blockDim.x) mt[k] = src[k];
    __syncthreads();

    auto mix = [](uint32_t u, uint32_t v) {
        return (((u & 0x80000000u) | (v & 0x7fffffffu)) >> 1) ^ ((0u - (v & 1u)) & 0x9908b0dfu);
    };

    for (int w = 0; w < MT_SEG_TWISTS; ++w) {
        uint32_t nv = 0;
        // stage 1: k in [0, 227)
        if (tid < MT_N - MT_M) nv = mt[tid + MT_M] ^ mix(mt[tid], mt[tid + 1]);
        __syncthreads();
        if (tid < MT_N - MT_M) mt[tid] = nv;
        __syncthreads();
        // stage 2: k in [227, 454)
        int k = (MT_N - MT_M) + tid;
        if (tid < MT_N - MT_M) nv = mt[k - (MT_N - MT_M)] ^ mix(mt[k], mt[k + 1]);
        __syncthreads();
        if (tid < MT_N - MT_M) mt[k] = nv;
        __syncthreads();
        // stage 3: k in [454, 624)
        k = 2 * (MT_N - MT_M) + tid;
        if (k < MT_N) nv = mt[k - (MT_N - MT_M)] ^ mix(mt[k], mt[(k + 1) % MT_N]);
        __syncthreads();
        if (k < MT_N) mt[k] = nv;
        __syncthreads();

        for (int q = tid; q < MT_N; q += blockDim.x) {
            uint32_t y = mt[q];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            float f = __uint2float_rn(y) * 2.3283064365386963e-10f;  // / 2^32 (exact)
            if (f >= 1.0f) f = 0.99999994f;
            dst[static_cast<size_t>(w) * MT_N + q] = f;
        }
    }
}

constexpr int KIJ_THREADS = 128;

// One CUDA block per row i. Threads sweep j in chunks; a block-wide scan assigns
// each non-culled pair its index in the sequential random-number stream.
__global__ void __launch_bounds__(KIJ_THREADS) formFactorKernel(
    const Triangle* __restrict__ tris, const float* __restrict__ areas,
    const DevNode* __restrict__ nodes, const LeafTri* __restrict__ leaves,
    const float4* __restrict__ rnd, unsigned long long scratchBaseNum,
    const unsigned long long* __restrict__ rowStart, int rowBegin, int n,
    Entry* __restrict__ mat, unsigned long long* __restrict__ nonZeroCount) {
    const int i = rowBegin + blockIdx.x;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wid = tid >> 5;
    const unsigned long long base = rowStart[i];

    const Triangle triI = tris[i];
    const Vec3 nI = triI.normal();

    __shared__ int warpCnt[KIJ_THREADS / 32];
    int carry = 0;
    unsigned localNz = 0;

    for (int chunk = 0; chunk < n; chunk += KIJ_THREADS) {
        const int j = chunk + tid;
        bool flag = false;
        Triangle triJ;
        if (j < n && j != i) {
            triJ = tris[j];
            flag = !(nI.dot(triJ.normal()) > 0.99f);
        }
        unsigned m = __ballot_sync(0xffffffffu, flag);
        if (lane == 0) warpCnt[wid] = __popc(m);
        __syncthreads();
        int off = carry + __popc(m & ((1u << lane) - 1u));
        int total = 0;
        #pragma unroll
        for (int w = 0; w < KIJ_THREADS / 32; ++w) {
            if (w < wid) off += warpCnt[w];
            total += warpCnt[w];
        }
        carry += total;

        if (flag) {
            const float4* r = rnd + ((64ull * (base + off) - scratchBaseNum) >> 2);
            const Vec3 nJ = triJ.normal();
            val_t kij = ZERO;

            for (int ray = 0; ray < NUM_RAYS; ++ray) {
                float4 q = r[ray];
                Vec3 pI = randomPointInTriangleD(triI, q.x, q.y);
                Vec3 pJ = randomPointInTriangleD(triJ, q.z, q.w);

                if (isRayBlockedD(pI, pJ, nodes, leaves, i, j)) continue;

                Vec3 v = pJ - pI;
                val_t distSqr = v.squaredNorm();
                if (distSqr < EPSILON) continue;

                val_t cosPhiI = cosPhiD(v, nI);
                val_t cosPhiJ = cosPhiD(-v, nJ);

                if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

                kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
            }
            kij = kij * INV_NUM_RAYS;

            if (kij > EPSILON) ++localNz;
            mat[static_cast<size_t>(j) * n + i].w = fminf(kij * areas[j], ONE);
        }
        __syncthreads();  // warpCnt reuse
    }
    if (localNz) atomicAdd(nonZeroCount, static_cast<unsigned long long>(localNz));
}

// One timestep of wave propagation; thread i sums over j in ascending order
__global__ void simStepKernel(const Entry* __restrict__ mat, const float* __restrict__ rho,
                              float* __restrict__ radB, int n, int t, int src, int timeOff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    val_t sumB = ZERO;
    for (int j = 0; j < n; ++j) {
        Entry e = mat[static_cast<size_t>(j) * n + i];
        if (e.w <= ZERO || t < e.tau) continue;
        sumB += e.w * radB[static_cast<size_t>(t - e.tau) * n + j];
    }
    val_t radE = (i == src && t < timeOff) ? 1.0f : ZERO;
    radB[static_cast<size_t>(t) * n + i] = rho[i] * sumB + radE;
}

// One block per triangle; threads handle lags t, serial summation per lag
__global__ void distanceKernel(const float* __restrict__ radB, float* __restrict__ distances,
                               int n, int steps, int src) {
    extern __shared__ float sums[];
    const int i = blockIdx.x;
    for (int t = threadIdx.x; t < steps; t += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = t; tt < steps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * n + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * n + src];
            sum += pS * pB;
        }
        sums[t] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        val_t maxCorr = ZERO;
        int bestT = 0;
        for (int t = 0; t < steps; ++t) {
            if (sums[t] > maxCorr) {
                maxCorr = sums[t];
                bestT = t;
            }
        }
        distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), copied back from device
    std::vector<val_t> distances;   // Computed distances from source
    unsigned long long nonZeroKij = 0;  // Number of form factors above EPSILON

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Device buffers
    Triangle* dTris = nullptr;
    float* dAreas = nullptr;
    float* dRho = nullptr;
    DevNode* dNodes = nullptr;
    LeafTri* dLeaves = nullptr;
    Entry* dMat = nullptr;          // Kij weights and Tau, transposed (N x N)
    float* dRadB = nullptr;
    float* dDist = nullptr;
};

// ============================================================================
// Initialization
// ============================================================================

template <typename T>
static T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    return p;
}

template <typename T>
static T* deviceUpload(const std::vector<T>& v) {
    T* p = deviceAlloc<T>(std::max<size_t>(v.size(), 1));
    if (!v.empty()) CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}

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

    // Flatten the octree and upload everything to the GPU
    std::vector<DevNode> nodes;
    std::vector<LeafTri> leaves;
    int maxDepth = 0;
    flattenOctree(state.octree, nodes, leaves, 0, maxDepth);
    if (7 * maxDepth + 1 > TRAVERSAL_STACK) {
        fprintf(stderr, "Octree too deep for traversal stack (%d)\n", maxDepth);
        exit(1);
    }

    CUDA_CHECK(cudaFree(0));
    state.dTris = deviceUpload(state.triangles);
    state.dAreas = deviceUpload(state.areas);
    state.dRho = deviceUpload(state.rho);
    state.dNodes = deviceUpload(nodes);
    state.dLeaves = deviceUpload(leaves);
    state.dMat = deviceAlloc<Entry>(state.numTriangles * state.numTriangles);
    size_t radSize = std::max<size_t>(timesteps * state.numTriangles, 1);
    state.dRadB = deviceAlloc<float>(radSize);
    CUDA_CHECK(cudaMemset(state.dRadB, 0, radSize * sizeof(float)));
    state.dDist = deviceAlloc<float>(state.numTriangles);
    CUDA_CHECK(cudaMemset(state.dDist, 0, state.numTriangles * sizeof(float)));
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    int n = static_cast<int>(state.numTriangles);
    dim3 grid(n, (n + 255) / 256);
    initMatrixKernel<<<grid, 256>>>(state.dTris, n, state.dMat);
    CUDA_CHECK(cudaGetLastError());
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int n = static_cast<int>(state.numTriangles);
    const size_t N = state.numTriangles;
    constexpr size_t MAX_BATCH_PAIRS = size_t(1) << 21;
    constexpr size_t SEG_NUMS = size_t(MT_SEG_TWISTS) * MT_N;

    // Per-row counts of pairs consuming random numbers -> stream offsets
    unsigned* dCounts = deviceAlloc<unsigned>(N);
    countPairsKernel<<<n, 128>>>(state.dTris, n, dCounts);
    CUDA_CHECK(cudaGetLastError());
    std::vector<unsigned> counts(N);
    CUDA_CHECK(cudaMemcpy(counts.data(), dCounts, N * sizeof(unsigned), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(dCounts));

    std::vector<unsigned long long> rowStart(N + 1, 0);
    for (size_t i = 0; i < N; ++i) rowStart[i + 1] = rowStart[i] + counts[i];
    unsigned long long* dRowStart = deviceUpload(rowStart);

    // Plan batches of consecutive rows
    struct Batch {
        int r0, r1;
        size_t blockStart, numSeg;
    };
    std::vector<Batch> batches;
    size_t maxSeg = 1;
    for (size_t r = 0; r < N;) {
        size_t r1 = r + 1;
        while (r1 < N && rowStart[r1 + 1] - rowStart[r] <= MAX_BATCH_PAIRS) ++r1;
        unsigned long long nb0 = rowStart[r] * 64, nb1 = rowStart[r1] * 64;
        if (nb1 > nb0) {
            size_t b0 = static_cast<size_t>(nb0 / MT_N) / MT_SEG_TWISTS * MT_SEG_TWISTS;
            size_t b1 = (static_cast<size_t>((nb1 + MT_N - 1) / MT_N) + MT_SEG_TWISTS - 1) /
                        MT_SEG_TWISTS * MT_SEG_TWISTS;
            batches.push_back({static_cast<int>(r), static_cast<int>(r1), b0,
                               (b1 - b0) / MT_SEG_TWISTS});
            maxSeg = std::max(maxSeg, (b1 - b0) / MT_SEG_TWISTS);
        }
        r = r1;
    }

    float* dScratch = deviceAlloc<float>(maxSeg * SEG_NUMS);
    uint32_t* dCp = deviceAlloc<uint32_t>(maxSeg * MT_N);
    uint32_t* hCp[2];
    cudaEvent_t uploaded[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&hCp[b], maxSeg * MT_N * sizeof(uint32_t)));
        CUDA_CHECK(cudaEventCreateWithFlags(&uploaded[b], cudaEventDisableTiming));
    }
    unsigned long long* dNz = deviceAlloc<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(dNz, 0, sizeof(unsigned long long)));

    // Sequential MT19937 state, advanced only to the checkpoints needed
    std::vector<uint32_t> cur(MT_N);
    mtSeed(cur.data(), 42);
    size_t curBlock = 0;

    for (size_t b = 0; b < batches.size(); ++b) {
        const Batch& bt = batches[b];
        const int buf = static_cast<int>(b & 1);
        if (b >= 2) CUDA_CHECK(cudaEventSynchronize(uploaded[buf]));

        for (size_t s = 0; s < bt.numSeg; ++s) {
            size_t target = bt.blockStart + s * MT_SEG_TWISTS;
            while (curBlock < target) {
                mtTwist(cur.data());
                ++curBlock;
            }
            memcpy(hCp[buf] + s * MT_N, cur.data(), MT_N * sizeof(uint32_t));
        }

        CUDA_CHECK(cudaMemcpyAsync(dCp, hCp[buf], bt.numSeg * MT_N * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaEventRecord(uploaded[buf]));
        mtGenKernel<<<static_cast<unsigned>(bt.numSeg), 256>>>(dCp, dScratch);
        CUDA_CHECK(cudaGetLastError());
        formFactorKernel<<<bt.r1 - bt.r0, KIJ_THREADS>>>(
            state.dTris, state.dAreas, state.dNodes, state.dLeaves,
            reinterpret_cast<const float4*>(dScratch),
            static_cast<unsigned long long>(bt.blockStart) * MT_N, dRowStart, bt.r0, n,
            state.dMat, dNz);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(&state.nonZeroKij, dNz, sizeof(unsigned long long), cudaMemcpyDeviceToHost));

    for (size_t i = 0; i < N; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == N) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, N);
        }
    }

    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFreeHost(hCp[b]));
        CUDA_CHECK(cudaEventDestroy(uploaded[b]));
    }
    CUDA_CHECK(cudaFree(dScratch));
    CUDA_CHECK(cudaFree(dCp));
    CUDA_CHECK(cudaFree(dRowStart));
    CUDA_CHECK(cudaFree(dNz));
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int timeOff = static_cast<int>(state.numTimesteps / 2);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simStepKernel<<<(n + 127) / 128, 128>>>(state.dMat, state.dRho, state.dRadB, n,
                                                static_cast<int>(t),
                                                static_cast<int>(state.sourceIndex), timeOff);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    state.radB.resize(state.numTimesteps * state.numTriangles);
    if (!state.radB.empty()) {
        CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, state.radB.size() * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int n = static_cast<int>(state.numTriangles);
    const int steps = static_cast<int>(state.numTimesteps);

    if (steps > 0) {
        distanceKernel<<<n, 64, steps * sizeof(float)>>>(state.dRadB, state.dDist, n, steps,
                                                        static_cast<int>(state.sourceIndex));
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.dDist, state.numTriangles * sizeof(float),
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
    // Load all kernels at startup so lazy module loading does not land in timed regions
    setenv("CUDA_MODULE_LOADING", "EAGER", 0);

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
        if (!validateResults(state)) {
            return 1;
        }
    }

    return 0;
}
