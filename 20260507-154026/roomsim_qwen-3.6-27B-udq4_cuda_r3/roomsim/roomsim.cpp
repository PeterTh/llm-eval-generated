/**
 * Room Response Simulation Benchmark
 * 
 * CUDA-parallelized implementation of room impulse response simulation using
 * radiosity-based wave propagation. Models how sound/light waves propagate
 * between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 * All compute-intensive phases are parallelized on the GPU.
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
#include <cfloat>

#include "../common/results_output.hpp"
#include <cuda_runtime.h>

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
// Vector and Triangle Types (host)
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
// Device data structures (POD, __align__ for CUDA)
// ============================================================================

struct __align__(16) DVec3 {
    float x, y, z;
};

struct __align__(64) DTriangle {
    DVec3 a, b, c, _normal;
};

// Flattened octree node for GPU traversal
struct OctreeNode {
    float center[3];
    float halfExtent[3];
    int childStart;   // index into triangleIndices array, or -1 if internal
    int childCount;   // number of triangle indices in leaf, or 8 for internal
    int children[8];  // node indices of children (for internal nodes)
    int isLeaf;       // 1 = leaf, 0 = internal
    int triCount;     // number of triangles stored in this leaf
};

// ============================================================================
// Triangle-Box Overlap Test (for octree construction on host)
// ============================================================================

bool triangleBoxOverlap(const Vec3& boxCenter, const Vec3& boxHalfSize, const Triangle& tri) {
    Vec3 v0 = tri.a - boxCenter;
    Vec3 v1 = tri.b - boxCenter;
    Vec3 v2 = tri.c - boxCenter;

    Vec3 e0 = v1 - v0;
    Vec3 e1 = v2 - v1;
    Vec3 e2 = v0 - v2;

    auto minMax3 = [](val_t a, val_t b, val_t c) {
        return std::make_pair(std::min({a, b, c}), std::max({a, b, c}));
    };

    auto [minX, maxX] = minMax3(v0.x, v1.x, v2.x);
    if (minX > boxHalfSize.x || maxX < -boxHalfSize.x) return false;

    auto [minY, maxY] = minMax3(v0.y, v1.y, v2.y);
    if (minY > boxHalfSize.y || maxY < -boxHalfSize.y) return false;

    auto [minZ, maxZ] = minMax3(v0.z, v1.z, v2.z);
    if (minZ > boxHalfSize.z || maxZ < -boxHalfSize.z) return false;

    Vec3 triNormal = e0.cross(e1);
    val_t d = triNormal.dot(v0);
    val_t r = boxHalfSize.x * std::abs(triNormal.x) +
              boxHalfSize.y * std::abs(triNormal.y) +
              boxHalfSize.z * std::abs(triNormal.z);
    if (std::abs(d) > r) return false;

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
// Octree for Spatial Acceleration (host-side construction)
// ============================================================================

constexpr size_t MAX_OCTREE_TRIS = 8;
constexpr val_t MAX_OCTREE_LEAF_SIZE = 0.5f;

class Octree {
public:
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    std::unique_ptr<Octree> children[8];
    std::vector<size_t> triangleIndices;
    const std::vector<Triangle>* allTriangles;

    Octree() : allTriangles(nullptr) {}

    void build(const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        if (triangles.empty()) return;

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

        std::vector<size_t> allIndices(triangles.size());
        for (size_t i = 0; i < triangles.size(); ++i) allIndices[i] = i;

        buildNode(allIndices, minBound, maxBound);
    }

    // Flatten octree into linear arrays for GPU
    void flatten(std::vector<OctreeNode>& nodes, std::vector<int>& triIndices) const {
        nodes.clear();
        triIndices.clear();
        flattenNode(*this, nodes, triIndices);
    }

private:
    void buildNode(const std::vector<size_t>& indices, const Vec3& nodeMin, const Vec3& nodeMax) {
        minBound = nodeMin;
        maxBound = nodeMax;
        halfExtent = (maxBound - minBound) * 0.5f;
        center = (minBound + maxBound) * 0.5f;

        if (indices.size() <= MAX_OCTREE_TRIS ||
            (maxBound - minBound).norm() < MAX_OCTREE_LEAF_SIZE) {
            triangleIndices = indices;
            return;
        }

        Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];

        for (size_t idx : indices) {
            const Triangle& tri = (*allTriangles)[idx];
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

    int flattenNode(const Octree& node, std::vector<OctreeNode>& nodes, std::vector<int>& triIndices) const {
        int myIdx = static_cast<int>(nodes.size());
        OctreeNode on;
        on.center[0] = node.center.x;
        on.center[1] = node.center.y;
        on.center[2] = node.center.z;
        on.halfExtent[0] = node.halfExtent.x;
        on.halfExtent[1] = node.halfExtent.y;
        on.halfExtent[2] = node.halfExtent.z;

        if (!node.triangleIndices.empty()) {
            on.isLeaf = 1;
            on.triCount = static_cast<int>(node.triangleIndices.size());
            on.childStart = static_cast<int>(triIndices.size());
            on.childCount = 0;
            for (auto idx : node.triangleIndices) {
                triIndices.push_back(static_cast<int>(idx));
            }
            memset(on.children, -1, sizeof(on.children));
        } else {
            on.isLeaf = 0;
            on.triCount = 0;
            on.childStart = -1;
            on.childCount = 0;
            for (int i = 0; i < 8; ++i) {
                if (node.children[i]) {
                    on.children[i] = flattenNode(*node.children[i], nodes, triIndices);
                    on.childCount++;
                } else {
                    on.children[i] = -1;
                }
            }
        }
        nodes.push_back(on);
        return myIdx;
    }
};

// ============================================================================
// Mesh Generation: Icosphere
// ============================================================================

class IcosphereMesh {
public:
    std::vector<Triangle> triangles;

    IcosphereMesh(int subdivisions, val_t radius) {
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

        std::vector<std::array<idx_t, 3>> faces = {
            {0, 11, 5}, {0, 5, 1}, {0, 1, 7}, {0, 7, 10}, {0, 10, 11},
            {1, 5, 9}, {5, 11, 4}, {11, 10, 2}, {10, 7, 6}, {7, 1, 8},
            {3, 9, 4}, {3, 4, 2}, {3, 2, 6}, {3, 6, 8}, {3, 8, 9},
            {4, 9, 5}, {2, 4, 11}, {6, 2, 10}, {8, 6, 7}, {9, 8, 1}
        };

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

        triangles.reserve(faces.size());
        for (const auto& face : faces) {
            triangles.emplace_back(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;
    std::vector<val_t> rho;
    std::vector<val_t> kij;
    std::vector<int> tau;
    std::vector<val_t> radE;
    std::vector<val_t> radB;
    std::vector<val_t> distances;

    Octree octree;

    size_t sourceIndex;

    size_t idx2d(size_t i, size_t j) const { return i * numTriangles + j; }
    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

// ============================================================================
// CUDA Kernels
// ============================================================================

// xorshift32 PRNG (device)
__device__ __forceinline__ uint32_t xorshift32(uint32_t& state) {
    uint32_t x = state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    state = x;
    return x;
}

__device__ __forceinline__ float xorshift32f(uint32_t& state) {
    return static_cast<float>(xorshift32(state)) / 4294967296.0f;
}

__device__ __forceinline__ DVec3 d_vec3_add(DVec3 a, DVec3 b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
__device__ __forceinline__ DVec3 d_vec3_sub(DVec3 a, DVec3 b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
__device__ __forceinline__ DVec3 d_vec3_mul(DVec3 a, float s) {
    return {a.x * s, a.y * s, a.z * s};
}
__device__ __forceinline__ DVec3 d_vec3_div(DVec3 a, float s) {
    return {a.x / s, a.y / s, a.z / s};
}
__device__ __forceinline__ float d_vec3_dot(DVec3 a, DVec3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
__device__ __forceinline__ DVec3 d_vec3_cross(DVec3 a, DVec3 b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}
__device__ __forceinline__ float d_vec3_sqnorm(DVec3 a) {
    return a.x * a.x + a.y * a.y + a.z * a.z;
}
__device__ __forceinline__ DVec3 d_vec3_normalize(DVec3 a) {
    float n = sqrtf(d_vec3_sqnorm(a));
    if (n > EPSILON) return d_vec3_div(a, n);
    return {0.0f, 0.0f, 0.0f};
}
__device__ __forceinline__ DVec3 d_vec3_center(const DTriangle& tri) {
    return d_vec3_div(d_vec3_add(d_vec3_add(tri.a, tri.b), tri.c), 3.0f);
}

// Ray-triangle intersection (Möller-Trumbore)
__device__ __forceinline__ float d_ray_tri_intersect(
    DVec3 orig, DVec3 dir, DVec3 v0, DVec3 v1, DVec3 v2) {
    DVec3 e1 = d_vec3_sub(v1, v0);
    DVec3 e2 = d_vec3_sub(v2, v0);
    DVec3 pvec = d_vec3_cross(dir, e2);
    float det = d_vec3_dot(e1, pvec);

    if (fabsf(det) < EPSILON) return FLT_MAX;

    float invDet = 1.0f / det;
    DVec3 tvec = d_vec3_sub(orig, v0);
    float u = d_vec3_dot(tvec, pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return FLT_MAX;

    DVec3 qvec = d_vec3_cross(tvec, e1);
    float v = d_vec3_dot(dir, qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return FLT_MAX;

    return d_vec3_dot(e2, qvec) * invDet;
}

// Octree ray-box intersection test
__device__ __forceinline__ bool d_ray_intersects_box(
    const OctreeNode& node, DVec3 p1, DVec3 p2) {
    DVec3 d = d_vec3_mul(d_vec3_sub(p2, p1), 0.5f);
    DVec3 c = d_vec3_sub(d_vec3_add(p1, d), {node.center[0], node.center[1], node.center[2]});
    DVec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
    float he[3] = {node.halfExtent[0], node.halfExtent[1], node.halfExtent[2]};

    if (fabsf(c.x) > he[0] + ad.x) return false;
    if (fabsf(c.y) > he[1] + ad.y) return false;
    if (fabsf(c.z) > he[2] + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) > he[1] * ad.z + he[2] * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) > he[2] * ad.x + he[0] * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) > he[0] * ad.y + he[1] * ad.x + EPSILON) return false;

    return true;
}

// Check if ray is blocked by traversing octree (stack-based)
__device__ __forceinline__ bool d_is_ray_blocked(
    DVec3 from, DVec3 to,
    const OctreeNode* nodes, const int* triIndices, int rootIdx,
    int srcTriIdx, int dstTriIdx) {
    float rayLenSq = d_vec3_sqnorm(d_vec3_sub(to, from));
    if (rayLenSq < EPSILON) return true;
    float rayLen = sqrtf(rayLenSq);
    DVec3 dirNorm = d_vec3_div(d_vec3_sub(to, from), rayLen);

    // Stack-based traversal (max depth ~20 for our octrees)
    int stack[32];
    int sp = 0;
    stack[sp++] = rootIdx;

    while (sp > 0) {
        int idx = stack[--sp];
        const OctreeNode& node = nodes[idx];

        if (!d_ray_intersects_box(node, from, to)) continue;

        if (node.isLeaf) {
            int start = node.childStart;
            int count = node.triCount;
            for (int k = 0; k < count; ++k) {
                int ti = triIndices[start + k];
                if (ti == srcTriIdx || ti == dstTriIdx) continue;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                if (node.children[c] >= 0) {
                    stack[sp++] = node.children[c];
                }
            }
        }
    }

    return false;
}

// Check if ray is blocked - checks actual triangle intersections
__device__ __forceinline__ bool d_is_ray_blocked_check(
    DVec3 from, DVec3 to,
    const DTriangle* triangles,
    const OctreeNode* nodes, const int* triIndices, int rootIdx,
    int srcTriIdx, int dstTriIdx) {
    float rayLenSq = d_vec3_sqnorm(d_vec3_sub(to, from));
    if (rayLenSq < EPSILON) return true;
    float rayLen = sqrtf(rayLenSq);
    DVec3 dirNorm = d_vec3_div(d_vec3_sub(to, from), rayLen);

    int stack[32];
    int sp = 0;
    stack[sp++] = rootIdx;

    while (sp > 0) {
        int idx = stack[--sp];
        const OctreeNode& node = nodes[idx];

        if (!d_ray_intersects_box(node, from, to)) continue;

        if (node.isLeaf) {
            int start = node.childStart;
            int count = node.triCount;
            for (int k = 0; k < count; ++k) {
                int ti = triIndices[start + k];
                if (ti == srcTriIdx || ti == dstTriIdx) continue;

                const DTriangle& tri = triangles[ti];
                float dist = d_ray_tri_intersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                if (node.children[c] >= 0) {
                    stack[sp++] = node.children[c];
                }
            }
        }
    }

    return false;
}

// CosPhi on device
__device__ __forceinline__ float d_cos_phi(DVec3 v, DVec3 normal) {
    float vNorm = sqrtf(d_vec3_sqnorm(v));
    if (vNorm <= EPSILON) return 0.0f;
    return fmaxf(0.0f, d_vec3_dot(v, normal) / vNorm);
}

// Random point in triangle on device
__device__ __forceinline__ DVec3 d_random_point_in_triangle(const DTriangle& tri, uint32_t& rngState) {
    float u = xorshift32f(rngState);
    float v = xorshift32f(rngState);
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    DVec3 ab = d_vec3_sub(tri.b, tri.a);
    DVec3 ac = d_vec3_sub(tri.c, tri.a);
    return d_vec3_add(tri.a, d_vec3_add(d_vec3_mul(ab, u), d_vec3_mul(ac, v)));
}

// ============================================================================
// Kernel: Compute Time Delays (Tau)
// ============================================================================

__global__ void compute_tau_kernel(
    const DTriangle* triangles,
    int* tau,
    int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    int i = idx / n;
    int j = idx % n;
    if (i == j) return;

    DVec3 ci = d_vec3_center(triangles[i]);
    DVec3 cj = d_vec3_center(triangles[j]);
    float dist = sqrtf(d_vec3_sqnorm(d_vec3_sub(ci, cj)));
    tau[idx] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Kernel: Compute Form Factors (Kij)
// ============================================================================

__global__ void compute_kij_kernel(
    const DTriangle* triangles,
    const OctreeNode* nodes,
    const int* triIndices,
    int rootIdx,
    float* kij,
    int n,
    uint32_t seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n * n) return;

    int i = idx / n;
    int j = idx % n;
    if (i == j) return;

    const DTriangle& triI = triangles[i];
    const DTriangle& triJ = triangles[j];

    // Cull triangles facing same direction
    if (d_vec3_dot(triI._normal, triJ._normal) > 0.99f) return;

    // Deterministic per-pair seed
    uint32_t rngState = seed ^ (static_cast<uint32_t>(i) * 2654435761u) ^ (static_cast<uint32_t>(j) * 2246822519u);
    if (rngState == 0) rngState = 1;

    float kij_val = 0.0f;

    for (int r = 0; r < NUM_RAYS; ++r) {
        DVec3 pI = d_random_point_in_triangle(triI, rngState);
        DVec3 pJ = d_random_point_in_triangle(triJ, rngState);

        if (d_is_ray_blocked_check(pI, pJ, triangles, nodes, triIndices, rootIdx, i, j)) continue;

        DVec3 v = d_vec3_sub(pJ, pI);
        float distSqr = d_vec3_sqnorm(v);
        if (distSqr < EPSILON) continue;

        float cosPhiI = d_cos_phi(v, triI._normal);
        float cosPhiJ = d_cos_phi(d_vec3_mul(v, -1.0f), triJ._normal);

        if (cosPhiI <= 0.0f || cosPhiJ <= 0.0f) continue;

        kij_val += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[idx] = kij_val * INV_NUM_RAYS;
}

// ============================================================================
// Kernel: Simulation timestep (parallel over triangles)
// ============================================================================

__global__ void simulation_kernel(
    const float* kij,
    const int* tau,
    const float* areas,
    const float* rho,
    const float* radE,
    float* radB,
    int n,
    int t,
    int timeOn,
    int timeOff) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float sumB = 0.0f;

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        int tauij = tau[i * n + j];

        if (t < tauij) continue;

        float kij_val = kij[i * n + j];
        if (kij_val <= 0.0f) continue;

        int srcTime = t - tauij;
        float radJ = radB[srcTime * n + j];
        if (radJ <= 0.0f) continue;

        sumB += fminf(kij_val * areas[j], 1.0f) * radJ;
    }

    radB[t * n + i] = rho[i] * sumB + radE[t * n + i];
}

// ============================================================================
// Kernel: Distance computation (parallel over triangles)
// ============================================================================

__global__ void distance_kernel(
    const float* radB,
    float* distances,
    int n,
    int timesteps,
    int sourceIndex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    float maxCorr = 0.0f;
    int bestT = 0;

    for (int t = 0; t < timesteps; ++t) {
        float sum = 0.0f;

        for (int tt = t; tt < timesteps; ++tt) {
            float pB = radB[tt * n + i];
            float pS = radB[(tt - t) * n + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = WAVE_SPEED * static_cast<float>(bestT);
}

// ============================================================================
// Initialization
// ============================================================================

void initializeSimulation(SimulationState& state, int subdivisions, size_t timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    IcosphereMesh mesh(subdivisions, 10.0f);
    state.triangles = std::move(mesh.triangles);
    state.numTriangles = state.triangles.size();
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;

    printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);

    printf("Building octree...\n");
    state.octree.build(state.triangles);

    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    state.rho.resize(state.numTriangles, reflectivity);

    state.kij.resize(state.numTriangles * state.numTriangles, ZERO);
    state.tau.resize(state.numTriangles * state.numTriangles, 0);
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }
}

// ============================================================================
// CUDA-accelerated compute phases
// ============================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            printf("CUDA error at %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(1); \
        } \
    } while(0)

void computeTimeDelaysCUDA(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    size_t n = state.numTriangles;
    size_t total = n * n;

    // Copy triangles to device
    DTriangle* d_triangles;
    CUDA_CHECK(cudaMalloc(&d_triangles, n * sizeof(DTriangle)));
    std::vector<DTriangle> h_triangles(n);
    for (size_t i = 0; i < n; ++i) {
        h_triangles[i].a = {state.triangles[i].a.x, state.triangles[i].a.y, state.triangles[i].a.z};
        h_triangles[i].b = {state.triangles[i].b.x, state.triangles[i].b.y, state.triangles[i].b.z};
        h_triangles[i].c = {state.triangles[i].c.x, state.triangles[i].c.y, state.triangles[i].c.z};
        h_triangles[i]._normal = {state.triangles[i]._normal.x, state.triangles[i]._normal.y, state.triangles[i]._normal.z};
    }
    CUDA_CHECK(cudaMemcpy(d_triangles, h_triangles.data(), n * sizeof(DTriangle), cudaMemcpyHostToDevice));

    // Copy tau to device
    int* d_tau;
    CUDA_CHECK(cudaMalloc(&d_tau, total * sizeof(int)));
    CUDA_CHECK(cudaMemset(d_tau, 0, total * sizeof(int)));

    int blockSize = 256;
    int gridSize = (static_cast<int>(total) + blockSize - 1) / blockSize;

    compute_tau_kernel<<<gridSize, blockSize>>>(d_triangles, d_tau, static_cast<int>(n));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy tau back
    CUDA_CHECK(cudaMemcpy(state.tau.data(), d_tau, total * sizeof(int), cudaMemcpyDeviceToHost));

    // Keep triangles on device for form factor computation
    // Flatten octree for GPU
    std::vector<OctreeNode> nodes;
    std::vector<int> triIndices;
    state.octree.flatten(nodes, triIndices);

    OctreeNode* d_nodes;
    CUDA_CHECK(cudaMalloc(&d_nodes, nodes.size() * sizeof(OctreeNode)));
    CUDA_CHECK(cudaMemcpy(d_nodes, nodes.data(), nodes.size() * sizeof(OctreeNode), cudaMemcpyHostToDevice));

    int* d_triIndices;
    CUDA_CHECK(cudaMalloc(&d_triIndices, triIndices.size() * sizeof(int)));
    CUDA_CHECK(cudaMemcpy(d_triIndices, triIndices.data(), triIndices.size() * sizeof(int), cudaMemcpyHostToDevice));

    // Allocate kij on device
    float* d_kij;
    CUDA_CHECK(cudaMalloc(&d_kij, total * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_kij, 0, total * sizeof(float)));

    // Compute form factors
    printf("Computing form factors (Kij)...\n");
    uint32_t seed = 42;
    compute_kij_kernel<<<gridSize, blockSize>>>(
        d_triangles, d_nodes, d_triIndices, 0, d_kij, static_cast<int>(n), seed);
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy kij back
    CUDA_CHECK(cudaMemcpy(state.kij.data(), d_kij, total * sizeof(float), cudaMemcpyDeviceToHost));

    printf("  Progress: %zu/%zu triangles\n", n, n);

    // Now run simulation on GPU
    printf("Running wave propagation simulation...\n");

    float* d_areas;
    CUDA_CHECK(cudaMalloc(&d_areas, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_areas, state.areas.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    float* d_rho;
    CUDA_CHECK(cudaMalloc(&d_rho, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_rho, state.rho.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    float* d_radE;
    size_t radSize = state.numTimesteps * n;
    CUDA_CHECK(cudaMalloc(&d_radE, radSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_radE, state.radE.data(), radSize * sizeof(float), cudaMemcpyHostToDevice));

    float* d_radB;
    CUDA_CHECK(cudaMalloc(&d_radB, radSize * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_radB, 0, radSize * sizeof(float)));

    int triBlockSize = 256;
    int triGridSize = (static_cast<int>(n) + triBlockSize - 1) / triBlockSize;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulation_kernel<<<triGridSize, triBlockSize>>>(
            d_kij, d_tau, d_areas, d_rho, d_radE, d_radB,
            static_cast<int>(n), static_cast<int>(t), 0, static_cast<int>(state.numTimesteps / 2));
        CUDA_CHECK(cudaDeviceSynchronize());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    // Copy radB back
    CUDA_CHECK(cudaMemcpy(state.radB.data(), d_radB, radSize * sizeof(float), cudaMemcpyDeviceToHost));

    // Compute distances on GPU
    printf("Computing distances via cross-correlation...\n");

    float* d_distances;
    CUDA_CHECK(cudaMalloc(&d_distances, n * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_distances, 0, n * sizeof(float)));

    distance_kernel<<<triGridSize, triBlockSize>>>(
        d_radB, d_distances, static_cast<int>(n), static_cast<int>(state.numTimesteps),
        static_cast<int>(state.sourceIndex));
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), d_distances, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Cleanup
    CUDA_CHECK(cudaFree(d_triangles));
    CUDA_CHECK(cudaFree(d_tau));
    CUDA_CHECK(cudaFree(d_kij));
    CUDA_CHECK(cudaFree(d_nodes));
    CUDA_CHECK(cudaFree(d_triIndices));
    CUDA_CHECK(cudaFree(d_areas));
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_radE));
    CUDA_CHECK(cudaFree(d_radB));
    CUDA_CHECK(cudaFree(d_distances));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state) {
    printf("\nValidation:\n");

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

    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

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

    // All computation on GPU
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelaysCUDA(state);

    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    printf("Precomputation time: %ld ms\n", preDuration);
    printf("\n");

    // Total time (simulation and distance are included in the CUDA call)
    long totalTime = preDuration;
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
