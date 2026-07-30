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

#include "../common/results_output.hpp"

// CUDA
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr int NUM_RAYS = 16;              // Number of rays for visibility sampling
constexpr val_t EPSILON = 1e-6f;

// ============================================================================
// Vector and Triangle Types
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
// GPU-friendly data structures
// ============================================================================

// Triangle representation suitable for GPU struct-of-arrays style access
struct alignas(16) GPUTriangle {
    float3 a, b, c, n;
};

// Flattened octree node for GPU access
struct FlatOctreeNode {
    float minX, minY, minZ, maxX, maxY, maxZ;
    int children[8];   // indices into flat node array, -1 = empty
    int firstTriIdx;   // index into flat triangle indices array (leaf only)
    int numTris;       // number of triangles in this node (0 for internal nodes)
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
// Octree Flattening for GPU
// ============================================================================

// Recursively flatten the CPU octree into a GPU-friendly flat node array
// Returns the index of the node in flatNodes
static int flattenNode(const Octree& node,
                       std::vector<FlatOctreeNode>& flatNodes,
                       std::vector<int>& flatTriIndices) {
    int idx = static_cast<int>(flatNodes.size());
    FlatOctreeNode fn;
    fn.minX = node.minBound.x; fn.minY = node.minBound.y; fn.minZ = node.minBound.z;
    fn.maxX = node.maxBound.x; fn.maxY = node.maxBound.y; fn.maxZ = node.maxBound.z;

    if (node.triangleIndices.empty()) {
        // Internal node: process children
        fn.firstTriIdx = 0;
        fn.numTris = 0;
        for (int i = 0; i < 8; ++i) fn.children[i] = -1;
        flatNodes.push_back(fn);
        FlatOctreeNode& placed = flatNodes[idx];
        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                placed.children[i] = flattenNode(*node.children[i], flatNodes, flatTriIndices);
            }
        }
    } else {
        // Leaf node: store triangle indices
        fn.firstTriIdx = static_cast<int>(flatTriIndices.size());
        fn.numTris = static_cast<int>(node.triangleIndices.size());
        for (int i = 0; i < 8; ++i) fn.children[i] = -1;
        flatNodes.push_back(fn);
        for (size_t i = 0; i < node.triangleIndices.size(); ++i) {
            flatTriIndices.push_back(static_cast<int>(node.triangleIndices[i]));
        }
    }
    return idx;
}

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
// Ray-Triangle Intersection (Möller-Trumbore algorithm) - CPU version
// ============================================================================

val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return std::numeric_limits<val_t>::max();

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return std::numeric_limits<val_t>::max();

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return std::numeric_limits<val_t>::max();

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// CUDA Device Functions and Kernels
// ============================================================================

// Xor-shift random number generator for GPU (returns float in [0, 1))
__device__ float d_rand(uint32_t& state) {
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return (float)(state & 0x7fffffffu) / 2147483648.0f;
}

__device__ float3 d_cross(float3 a, float3 b) {
    return make_float3(a.y * b.z - a.z * b.y,
                       a.z * b.x - a.x * b.z,
                       a.x * b.y - a.y * b.x);
}

__device__ float d_dot(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__device__ float d_cos_phi(float3 v, float3 normal) {
    float vNorm = sqrtf(d_dot(v, v));
    if (vNorm <= EPSILON) return 0.0f;
    return fmaxf(0.0f, d_dot(v, normal) / vNorm);
}

__device__ float3 d_random_point_in_triangle(const GPUTriangle& tri, uint32_t& seed) {
    float u = d_rand(seed);
    float v = d_rand(seed);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    return make_float3(
        tri.a.x + (tri.b.x - tri.a.x) * u + (tri.c.x - tri.a.x) * v,
        tri.a.y + (tri.b.y - tri.a.y) * u + (tri.c.y - tri.a.y) * v,
        tri.a.z + (tri.b.z - tri.a.z) * u + (tri.c.z - tri.a.z) * v
    );
}

// Möller-Trumbore ray-triangle intersection on GPU
// Returns true if hit and sets t to the distance
__device__ bool d_ray_triangle_intersect(float3 orig, float3 dir,
                                         float3 v0, float3 v1, float3 v2,
                                         float& t) {
    float3 e1 = make_float3(v1.x - v0.x, v1.y - v0.y, v1.z - v0.z);
    float3 e2 = make_float3(v2.x - v0.x, v2.y - v0.y, v2.z - v0.z);
    float3 pvec = d_cross(dir, e2);
    float det = d_dot(e1, pvec);

    if (fabsf(det) < EPSILON) return false;

    float invDet = 1.0f / det;
    float3 tvec = make_float3(orig.x - v0.x, orig.y - v0.y, orig.z - v0.z);
    float u = d_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return false;

    float3 qvec = d_cross(tvec, e1);
    float v = d_dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return false;

    t = d_dot(e2, qvec) * invDet;
    return true;
}

// Check if a ray segment intersects a box (for octree traversal on GPU)
__device__ bool d_ray_intersects_box(float3 p1, float3 p2,
                                     float3 boxMin, float3 boxMax) {
    float3 d = make_float3((p2.x - p1.x) * 0.5f,
                           (p2.y - p1.y) * 0.5f,
                           (p2.z - p1.z) * 0.5f);
    float3 center = make_float3((boxMin.x + boxMax.x) * 0.5f,
                                (boxMin.y + boxMax.y) * 0.5f,
                                (boxMin.z + boxMax.z) * 0.5f);
    float3 c = make_float3(p1.x + d.x - center.x,
                           p1.y + d.y - center.y,
                           p1.z + d.z - center.z);
    float3 halfExtent = make_float3((boxMax.x - boxMin.x) * 0.5f,
                                     (boxMax.y - boxMin.y) * 0.5f,
                                     (boxMax.z - boxMin.z) * 0.5f);
    float3 ad = make_float3(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    if (fabsf(c.x) > halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > halfExtent.z + ad.z) return false;

    float eps = 1e-6f;
    if (fabsf(d.y * c.z - d.z * c.y) > halfExtent.y * ad.z + halfExtent.z * ad.y + eps) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > halfExtent.z * ad.x + halfExtent.x * ad.z + eps) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > halfExtent.x * ad.y + halfExtent.y * ad.x + eps) return false;

    return true;
}

// Recursive octree traversal on GPU to check if ray is blocked
__device__ bool d_is_ray_blocked(float3 from, float3 to,
                                  const FlatOctreeNode* nodes,
                                  const int* triIndices,
                                  const GPUTriangle* triangles,
                                  int srcTriIdx, int dstTriIdx,
                                  int nodeIdx) {
    const FlatOctreeNode& node = nodes[nodeIdx];
    float3 boxMin = make_float3(node.minX, node.minY, node.minZ);
    float3 boxMax = make_float3(node.maxX, node.maxY, node.maxZ);

    if (!d_ray_intersects_box(from, to, boxMin, boxMax)) return false;

    if (node.numTris > 0) {
        // Leaf node: check individual triangles
        float3 dir = make_float3(to.x - from.x, to.y - from.y, to.z - from.z);
        float rayLen = sqrtf(d_dot(dir, dir));
        if (rayLen < EPSILON) return true;
        float3 dirNorm = make_float3(dir.x / rayLen, dir.y / rayLen, dir.z / rayLen);

        for (int t = 0; t < node.numTris; ++t) {
            int triIdx = triIndices[node.firstTriIdx + t];
            if (triIdx == srcTriIdx || triIdx == dstTriIdx) continue;

            const GPUTriangle& tri = triangles[triIdx];
            float hitT;
            if (d_ray_triangle_intersect(from, dirNorm, tri.a, tri.b, tri.c, hitT)) {
                if (hitT > EPSILON && hitT < rayLen - EPSILON) return true;
            }
        }
        return false;
    }

    // Internal node: descend to children
    for (int ci = 0; ci < 8; ++ci) {
        if (node.children[ci] >= 0) {
            if (d_is_ray_blocked(from, to, nodes, triIndices, triangles,
                                 srcTriIdx, dstTriIdx, node.children[ci]))
                return true;
        }
    }
    return false;
}

// ============================================================================
// CUDA Kernels
// ============================================================================

__global__ void computeTimeDelaysKernel(const GPUTriangle* triangles,
                                         int* tau, int N) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N || i == j) return;

    float3 ci = make_float3(
        (triangles[i].a.x + triangles[i].b.x + triangles[i].c.x) / 3.0f,
        (triangles[i].a.y + triangles[i].b.y + triangles[i].c.y) / 3.0f,
        (triangles[i].a.z + triangles[i].b.z + triangles[i].c.z) / 3.0f);
    float3 cj = make_float3(
        (triangles[j].a.x + triangles[j].b.x + triangles[j].c.x) / 3.0f,
        (triangles[j].a.y + triangles[j].b.y + triangles[j].c.y) / 3.0f,
        (triangles[j].a.z + triangles[j].b.z + triangles[j].c.z) / 3.0f);
    float dx = ci.x - cj.x, dy = ci.y - cj.y, dz = ci.z - cj.z;
    float dist = sqrtf(dx * dx + dy * dy + dz * dz);
    tau[i * N + j] = (int)ceilf(dist * (1.0f / WAVE_SPEED));
}

__global__ void computeFormFactorsKernel(const GPUTriangle* d_triangles,
                                          const FlatOctreeNode* d_nodes,
                                          const int* d_triIndices,
                                          float* d_kij, int N,
                                          uint32_t baseSeed) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N || j >= N || i == j) {
        if (i < N && j < N) d_kij[i * N + j] = 0.0f;
        return;
    }

    // Per-thread RNG seeded deterministically from (i, j)
    uint32_t seed = baseSeed ^ (uint32_t)(i * N + j) * 0x9e3779b9u;

    const GPUTriangle& triI = d_triangles[i];
    const GPUTriangle& triJ = d_triangles[j];

    // Cull triangles facing the same direction
    float ndot = d_dot(triI.n, triJ.n);
    if (ndot > 0.99f) { d_kij[i * N + j] = 0.0f; return; }

    float kij = 0.0f;

    for (int r = 0; r < NUM_RAYS; ++r) {
        float3 pI = d_random_point_in_triangle(triI, seed);
        float3 pJ = d_random_point_in_triangle(triJ, seed);

        if (d_is_ray_blocked(pI, pJ, d_nodes, d_triIndices, d_triangles,
                             i, j, 0))
            continue;

        float3 v = make_float3(pJ.x - pI.x, pJ.y - pI.y, pJ.z - pI.z);
        float distSqr = d_dot(v, v);
        if (distSqr < EPSILON) continue;

        float cosPhiI = d_cos_phi(v, triI.n);
        float cosPhiJ = d_cos_phi(make_float3(-v.x, -v.y, -v.z), triJ.n);

        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    d_kij[i * N + j] = kij * (1.0f / NUM_RAYS);
}

__global__ void simulationStepKernel(const float* d_kij, const int* d_tau,
                                      const float* d_areas,
                                      const float* d_rho,
                                      const float* d_radE_row,
                                      const float* d_radB,
                                      float* d_radB_row,
                                      int N, int t) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float sumB = 0.0f;

    for (int j = 0; j < N; ++j) {
        if (i == j) continue;

        int tauij = d_tau[i * N + j];
        if (t < tauij) continue;

        float kij = d_kij[i * N + j];
        if (kij <= 0.0f) continue;

        int srcTime = t - tauij;
        float radJ = d_radB[srcTime * N + j];
        if (radJ <= 0.0f) continue;

        sumB += fminf(kij * d_areas[j], 1.0f) * radJ;
    }

    d_radB_row[i] = d_rho[i] * sumB + d_radE_row[i];
}

__global__ void computeDistancesKernel(const float* d_radB,
                                        float* d_distances,
                                        int N, int T, int sourceIdx,
                                        float waveSpeed) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int t = 0; t < T; ++t) {
        float sumVal = 0.0f;
        for (int tt = t; tt < T; ++tt) {
            // Cross-correlation: shift source signal by t
            sumVal += d_radB[(tt - t) * N + sourceIdx] * d_radB[tt * N + i];
        }

        if (sumVal > maxCorr) {
            maxCorr = sumVal;
            bestT = t;
        }
    }

    d_distances[i] = waveSpeed * (float)bestT;
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
    std::vector<val_t> kij;         // Form factors (N x N matrix, row-major)
    std::vector<int> tau;           // Time delays (N x N matrix, row-major)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// GPU Resource Management
// ============================================================================

// Persistent GPU device pointers
static struct {
    GPUTriangle* d_triangles;
    FlatOctreeNode* d_octreeNodes;
    int* d_flatTriIndices;
    float* d_areas;
    float* d_rho;
    float* d_kij;
    int* d_tau;
    float* d_radE;
    float* d_radB;
    float* d_distances;
    int numOctreeNodes;
    int numFlatTriIndices;
    bool initialized;
} g_gpu;

static void gpu_initialize(SimulationState& state) {
    if (g_gpu.initialized) return;

    size_t N = state.numTriangles;
    size_t T = state.numTimesteps;

    // Convert and upload triangles
    std::vector<GPUTriangle> hostTris(N);
    for (size_t k = 0; k < N; ++k) {
        hostTris[k].a = make_float3(state.triangles[k].a.x, state.triangles[k].a.y, state.triangles[k].a.z);
        hostTris[k].b = make_float3(state.triangles[k].b.x, state.triangles[k].b.y, state.triangles[k].b.z);
        hostTris[k].c = make_float3(state.triangles[k].c.x, state.triangles[k].c.y, state.triangles[k].c.z);
        hostTris[k].n = make_float3(state.triangles[k].normal().x, state.triangles[k].normal().y, state.triangles[k].normal().z);
    }
    cudaMalloc(&g_gpu.d_triangles, N * sizeof(GPUTriangle));
    cudaMemcpy(g_gpu.d_triangles, hostTris.data(), N * sizeof(GPUTriangle), cudaMemcpyHostToDevice);

    // Flatten octree and upload
    std::vector<FlatOctreeNode> hostNodes;
    std::vector<int> hostTriIndices;
    flattenNode(state.octree, hostNodes, hostTriIndices);
    g_gpu.numOctreeNodes = static_cast<int>(hostNodes.size());
    g_gpu.numFlatTriIndices = static_cast<int>(hostTriIndices.size());
    cudaMalloc(&g_gpu.d_octreeNodes, hostNodes.size() * sizeof(FlatOctreeNode));
    cudaMalloc(&g_gpu.d_flatTriIndices, hostTriIndices.size() * sizeof(int));
    cudaMemcpy(g_gpu.d_octreeNodes, hostNodes.data(), hostNodes.size() * sizeof(FlatOctreeNode), cudaMemcpyHostToDevice);
    cudaMemcpy(g_gpu.d_flatTriIndices, hostTriIndices.data(), hostTriIndices.size() * sizeof(int), cudaMemcpyHostToDevice);

    // Upload areas and rho
    cudaMalloc(&g_gpu.d_areas, N * sizeof(float));
    cudaMalloc(&g_gpu.d_rho, N * sizeof(float));
    cudaMemcpy(g_gpu.d_areas, state.areas.data(), N * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(g_gpu.d_rho, state.rho.data(), N * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate matrices
    cudaMalloc(&g_gpu.d_kij, N * N * sizeof(float));
    cudaMalloc(&g_gpu.d_tau, N * N * sizeof(int));
    cudaMalloc(&g_gpu.d_radE, T * N * sizeof(float));
    cudaMalloc(&g_gpu.d_radB, T * N * sizeof(float));
    cudaMalloc(&g_gpu.d_distances, N * sizeof(float));

    // Upload radE
    cudaMemcpy(g_gpu.d_radE, state.radE.data(), T * N * sizeof(float), cudaMemcpyHostToDevice);
    // Initialize radB to zero on device
    cudaMemset(g_gpu.d_radB, 0, T * N * sizeof(float));

    g_gpu.initialized = true;
}

static void gpu_cleanup() {
    cudaFree(g_gpu.d_triangles);       g_gpu.d_triangles = nullptr;
    cudaFree(g_gpu.d_octreeNodes);     g_gpu.d_octreeNodes = nullptr;
    cudaFree(g_gpu.d_flatTriIndices);  g_gpu.d_flatTriIndices = nullptr;
    cudaFree(g_gpu.d_areas);           g_gpu.d_areas = nullptr;
    cudaFree(g_gpu.d_rho);             g_gpu.d_rho = nullptr;
    cudaFree(g_gpu.d_kij);             g_gpu.d_kij = nullptr;
    cudaFree(g_gpu.d_tau);             g_gpu.d_tau = nullptr;
    cudaFree(g_gpu.d_radE);            g_gpu.d_radE = nullptr;
    cudaFree(g_gpu.d_radB);            g_gpu.d_radB = nullptr;
    cudaFree(g_gpu.d_distances);       g_gpu.d_distances = nullptr;
    g_gpu.initialized = false;
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

    // Initialize matrices
    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) on GPU...\n");

    int N = static_cast<int>(state.numTriangles);
    gpu_initialize(state);

    // Initialize kij to zero
    cudaMemset(g_gpu.d_kij, 0, (size_t)N * N * sizeof(float));

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    computeFormFactorsKernel<<<grid, block>>>(
        g_gpu.d_triangles,
        g_gpu.d_octreeNodes, g_gpu.d_flatTriIndices,
        g_gpu.d_kij, N, 42u);

    cudaMemcpy(state.kij.data(), g_gpu.d_kij,
               (size_t)N * N * sizeof(float), cudaMemcpyDeviceToHost);

    printf("  Completed %d x %d form factors\n", N, N);
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) on GPU...\n");

    int N = static_cast<int>(state.numTriangles);
    gpu_initialize(state);

    dim3 block(16, 16);
    dim3 grid((N + 15) / 16, (N + 15) / 16);

    computeTimeDelaysKernel<<<grid, block>>>(g_gpu.d_triangles, g_gpu.d_tau, N);

    cudaMemcpy(state.tau.data(), g_gpu.d_tau,
               (size_t)N * N * sizeof(int), cudaMemcpyDeviceToHost);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation on GPU...\n");

    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);
    gpu_initialize(state);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    // radB is already zero-initialized on device from gpu_initialize
    // Step through each timestep sequentially (data dependency across t)
    for (int t = 0; t < T; ++t) {
        simulationStepKernel<<<numBlocks, blockSize>>>(
            g_gpu.d_kij, g_gpu.d_tau, g_gpu.d_areas, g_gpu.d_rho,
            g_gpu.d_radE + (size_t)t * N,
            g_gpu.d_radB,
            g_gpu.d_radB + (size_t)t * N,
            N, t);

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %d/%d\n", t + 1, T);
        }
    }

    cudaMemcpy(state.radB.data(), g_gpu.d_radB,
               (size_t)T * N * sizeof(float), cudaMemcpyDeviceToHost);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation on GPU...\n");

    int N = static_cast<int>(state.numTriangles);
    int T = static_cast<int>(state.numTimesteps);
    gpu_initialize(state);

    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;

    computeDistancesKernel<<<numBlocks, blockSize>>>(
        g_gpu.d_radB, g_gpu.d_distances,
        N, T, static_cast<int>(state.sourceIndex), WAVE_SPEED);

    cudaMemcpy(state.distances.data(), g_gpu.d_distances,
               (size_t)N * sizeof(float), cudaMemcpyDeviceToHost);
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
            gpu_cleanup();
            return 1;
        }
    }

    // Free GPU resources
    gpu_cleanup();

    return 0;
}
