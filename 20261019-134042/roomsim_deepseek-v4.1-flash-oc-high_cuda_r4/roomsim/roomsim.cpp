/**
 * Room Response Simulation Benchmark
 *
 * CUDA implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light waves propagate
 * between surfaces in a room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 * All numerically intensive phases run on the GPU.  The mt19937 random stream
 * used by the form-factor sampling is generated on the host in exactly the same
 * order as the original program so that results are identical.
 */

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cfloat>
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
constexpr int RANDOMS_PER_PAIR = NUM_RAYS * 4; // two random points per ray, two values each

// Set before initialization when host-side validation is requested.
static bool g_validateResults = false;

#ifdef __CUDACC__
#define HD __host__ __device__ __forceinline__
#else
#define HD inline
#endif

static void cudaCheck(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error (%s): %s\n", what, cudaGetErrorString(err));
        std::exit(1);
    }
}

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

    HD bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
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
// GPU-friendly flattened octree
// ============================================================================

struct GpuNode {
    Vec3 minBound, maxBound;
    Vec3 halfExtent, center;
    int child[8];
    int triStart;
    int triCount;
};

struct FlatOctree {
    std::vector<GpuNode> nodes;
    std::vector<int> indices;

    int buildRec(const Octree& node) {
        int id = static_cast<int>(nodes.size());
        GpuNode g;
        g.minBound = node.minBound;
        g.maxBound = node.maxBound;
        g.halfExtent = node.halfExtent;
        g.center = node.center;
        for (int i = 0; i < 8; ++i) g.child[i] = -1;
        g.triStart = 0;
        g.triCount = 0;
        nodes.push_back(g);

        for (int i = 0; i < 8; ++i) {
            if (node.children[i]) {
                int cid = buildRec(*node.children[i]);
                nodes[id].child[i] = cid;
            }
        }

        if (!node.triangleIndices.empty()) {
            nodes[id].triStart = static_cast<int>(indices.size());
            nodes[id].triCount = static_cast<int>(node.triangleIndices.size());
            for (size_t idx : node.triangleIndices) {
                indices.push_back(static_cast<int>(idx));
            }
        }
        return id;
    }

    int build(const Octree& root) {
        nodes.clear();
        indices.clear();
        nodes.reserve(1024);
        indices.reserve(1024);
        return buildRec(root);
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
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed) {}
    // Equivalent to std::uniform_real_distribution<float>(0, 1)(rng) for
    // libstdc++: one 32-bit engine draw scaled by the exact power of two 2^32.
    val_t rand() { return static_cast<val_t>(rng()) / 4294967296.0f; }
};

// Generate a random point inside a triangle using barycentric coordinates
HD Vec3 randomPointInTriangleUV(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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

// ============================================================================
// Device Visibility Testing (flattened octree)
// ============================================================================

HD bool d_rayIntersectsBox(const GpuNode& node, const Vec3& p1, const Vec3& p2) {
    Vec3 d = (p2 - p1) * 0.5f;
    Vec3 c = p1 + d - node.center;
    Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};

    if (fabsf(c.x) > node.halfExtent.x + ad.x) return false;
    if (fabsf(c.y) > node.halfExtent.y + ad.y) return false;
    if (fabsf(c.z) > node.halfExtent.z + ad.z) return false;

    if (fabsf(d.y * c.z - d.z * c.y) >
        node.halfExtent.y * ad.z + node.halfExtent.z * ad.y + EPSILON) return false;
    if (fabsf(d.z * c.x - d.x * c.z) >
        node.halfExtent.z * ad.x + node.halfExtent.x * ad.z + EPSILON) return false;
    if (fabsf(d.x * c.y - d.y * c.x) >
        node.halfExtent.x * ad.y + node.halfExtent.y * ad.x + EPSILON) return false;

    return true;
}

HD bool d_isRayBlocked(const Vec3& from, const Vec3& to,
                       const GpuNode* __restrict__ nodes,
                       const int* __restrict__ triIndices,
                       const Triangle* __restrict__ triangles,
                       int root,
                       size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[64];
    int sp = 0;
    stack[sp++] = root;

    while (sp > 0) {
        int ni = stack[--sp];
        const GpuNode& node = nodes[ni];

        if (node.triCount > 0) {
            int start = node.triStart;
            int count = node.triCount;
            for (int k = 0; k < count; ++k) {
                size_t idx = static_cast<size_t>(triIndices[start + k]);
                if (idx == srcTriIdx || idx == dstTriIdx) continue;

                const Triangle& tri = triangles[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) {
                    return true;
                }
            }
        } else {
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
            for (int i = 0; i < 8; ++i) {
                int ch = node.child[i];
                if (ch >= 0 && d_rayIntersectsBox(nodes[ch], from, to)) {
                    stack[sp++] = ch;
                }
            }
        }
    }
    return false;
}

HD val_t d_cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return c > ZERO ? c : ZERO;
}

// ============================================================================
// CUDA kernels
// ============================================================================

// Compute all time delays (Tau) in parallel.
__global__ void tauKernel(const Triangle* __restrict__ triangles,
                          int* __restrict__ tau,
                          int n,
                          unsigned long long total) {
    unsigned long long id = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (id >= total) return;

    int i = static_cast<int>(id / static_cast<unsigned long long>(n));
    int j = static_cast<int>(id % static_cast<unsigned long long>(n));
    if (i == j) return;

    Vec3 ci = (triangles[i].a + triangles[i].b + triangles[i].c) / 3.0f;
    Vec3 cj = (triangles[j].a + triangles[j].b + triangles[j].c) / 3.0f;
    val_t dist = (ci - cj).norm();
    tau[id] = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

// Compute the form factor for one (receiver, emitter) pair.
__global__ void kijKernel(const Triangle* __restrict__ triangles,
                          const GpuNode* __restrict__ nodes,
                          const int* __restrict__ triIndices,
                          int root,
                          const unsigned char* __restrict__ computeMask,
                          const val_t* __restrict__ randValues,
                          val_t* __restrict__ kij,
                          int n,
                          int rowBase,
                          int rows) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * n;
    if (tid >= total) return;

    int r = tid / n;
    int j = tid - r * n;
    int i = rowBase + r;

    if (!computeMask[tid]) return;

    const Triangle& triI = triangles[i];
    const Triangle& triJ = triangles[j];
    const val_t* rp = randValues + static_cast<size_t>(tid) * RANDOMS_PER_PAIR;

    val_t result = ZERO;
    for (int ray = 0; ray < NUM_RAYS; ++ray) {
        val_t u = rp[4 * ray + 0];
        val_t v = rp[4 * ray + 1];
        Vec3 pI = randomPointInTriangleUV(triI, u, v);

        u = rp[4 * ray + 2];
        v = rp[4 * ray + 3];
        Vec3 pJ = randomPointInTriangleUV(triJ, u, v);

        if (d_isRayBlocked(pI, pJ, nodes, triIndices, triangles, root,
                           static_cast<size_t>(i), static_cast<size_t>(j))) continue;

        Vec3 dir = pJ - pI;
        val_t distSqr = dir.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = d_cosPhi(dir, triI._normal);
        val_t cosPhiJ = d_cosPhi(-dir, triJ._normal);

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        result += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    kij[static_cast<size_t>(i) * n + j] = result * INV_NUM_RAYS;
}

// Propagate radiosity for one timestep, one thread per triangle.
__global__ void simulationKernel(int t,
                                 int n,
                                 const val_t* __restrict__ kij,
                                 const int* __restrict__ tau,
                                 const val_t* radB,
                                 const val_t* __restrict__ radE,
                                 const val_t* __restrict__ rho,
                                 const val_t* __restrict__ areas,
                                 val_t* radBOut) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t sumB = ZERO;
    size_t row = static_cast<size_t>(i) * n;

    for (int j = 0; j < n; ++j) {
        if (i == j) continue;

        int tauij = tau[row + j];

        // Skip if wave hasn't yet propagated from j to i
        if (t < tauij) continue;

        val_t kijVal = kij[row + j];
        if (kijVal <= ZERO) continue;

        size_t srcTime = static_cast<size_t>(t) - static_cast<size_t>(tauij);
        val_t radJ = radB[srcTime * static_cast<size_t>(n) + j];
        if (radJ <= ZERO) continue;

        val_t term = kijVal * areas[j];
        if (term > ONE) term = ONE;
        sumB += term * radJ;
    }

    radBOut[static_cast<size_t>(t) * n + i] =
        rho[i] * sumB + radE[static_cast<size_t>(t) * n + i];
}

// Compute distances via cross-correlation, one thread per triangle.
__global__ void distanceKernel(const val_t* __restrict__ radB,
                               val_t* __restrict__ distances,
                               int n,
                               int timesteps,
                               int sourceIndex) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    val_t maxCorr = ZERO;
    int bestT = 0;

    for (int t = 0; t < timesteps; ++t) {
        val_t sum = ZERO;

        for (int tt = t; tt < timesteps; ++tt) {
            val_t pB = radB[static_cast<size_t>(tt) * n + i];
            val_t pS = radB[static_cast<size_t>(tt - t) * n + sourceIndex];
            sum += pS * pB;
        }

        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = t;
        }
    }

    distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
}

// ============================================================================
// GPU buffers
// ============================================================================

struct DeviceBuffers {
    Triangle* triangles = nullptr;
    GpuNode* nodes = nullptr;
    int* triIndices = nullptr;
    val_t* areas = nullptr;
    val_t* rho = nullptr;
    val_t* kij = nullptr;
    int* tau = nullptr;
    val_t* radE = nullptr;
    val_t* radB = nullptr;
    val_t* distances = nullptr;

    int root = 0;

    ~DeviceBuffers() {
        if (triangles) cudaFree(triangles);
        if (nodes) cudaFree(nodes);
        if (triIndices) cudaFree(triIndices);
        if (areas) cudaFree(areas);
        if (rho) cudaFree(rho);
        if (kij) cudaFree(kij);
        if (tau) cudaFree(tau);
        if (radE) cudaFree(radE);
        if (radB) cudaFree(radB);
        if (distances) cudaFree(distances);
    }
};

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
    DeviceBuffers dev;              // GPU-resident data

    size_t sourceIndex;

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

    // Flatten the octree and upload all static data to the GPU.
    FlatOctree flat;
    state.dev.root = flat.build(state.octree);

    const size_t n = state.numTriangles;
    const size_t t = state.numTimesteps;
    DeviceBuffers& d = state.dev;

    cudaCheck(cudaMalloc(&d.triangles, n * sizeof(Triangle)), "malloc triangles");
    cudaCheck(cudaMemcpy(d.triangles, state.triangles.data(), n * sizeof(Triangle),
                         cudaMemcpyHostToDevice), "copy triangles");

    cudaCheck(cudaMalloc(&d.nodes, flat.nodes.size() * sizeof(GpuNode)), "malloc nodes");
    cudaCheck(cudaMemcpy(d.nodes, flat.nodes.data(), flat.nodes.size() * sizeof(GpuNode),
                         cudaMemcpyHostToDevice), "copy nodes");

    cudaCheck(cudaMalloc(&d.triIndices, flat.indices.size() * sizeof(int)), "malloc indices");
    cudaCheck(cudaMemcpy(d.triIndices, flat.indices.data(), flat.indices.size() * sizeof(int),
                         cudaMemcpyHostToDevice), "copy indices");

    cudaCheck(cudaMalloc(&d.areas, n * sizeof(val_t)), "malloc areas");
    cudaCheck(cudaMemcpy(d.areas, state.areas.data(), n * sizeof(val_t),
                         cudaMemcpyHostToDevice), "copy areas");

    cudaCheck(cudaMalloc(&d.rho, n * sizeof(val_t)), "malloc rho");
    cudaCheck(cudaMemcpy(d.rho, state.rho.data(), n * sizeof(val_t),
                         cudaMemcpyHostToDevice), "copy rho");

    cudaCheck(cudaMalloc(&d.kij, n * n * sizeof(val_t)), "malloc kij");
    cudaCheck(cudaMemset(d.kij, 0, n * n * sizeof(val_t)), "memset kij");

    cudaCheck(cudaMalloc(&d.tau, n * n * sizeof(int)), "malloc tau");
    cudaCheck(cudaMemset(d.tau, 0, n * n * sizeof(int)), "memset tau");

    cudaCheck(cudaMalloc(&d.radE, t * n * sizeof(val_t)), "malloc radE");
    cudaCheck(cudaMemcpy(d.radE, state.radE.data(), t * n * sizeof(val_t),
                         cudaMemcpyHostToDevice), "copy radE");

    cudaCheck(cudaMalloc(&d.radB, t * n * sizeof(val_t)), "malloc radB");
    cudaCheck(cudaMemset(d.radB, 0, t * n * sizeof(val_t)), "memset radB");

    cudaCheck(cudaMalloc(&d.distances, n * sizeof(val_t)), "malloc distances");
    cudaCheck(cudaMemset(d.distances, 0, n * sizeof(val_t)), "memset distances");
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const size_t n = state.numTriangles;
    const unsigned long long total = static_cast<unsigned long long>(n) * n;
    const int block = 256;
    const unsigned long long grid = (total + block - 1) / block;

    tauKernel<<<static_cast<unsigned int>(grid), block>>>(
        state.dev.triangles, state.dev.tau, static_cast<int>(n), total);
    cudaCheck(cudaGetLastError(), "launch tau");
    cudaCheck(cudaDeviceSynchronize(), "tau");
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const size_t n = state.numTriangles;
    RandomGenerator rng(42);

    // Two-stage pipeline: the host generates the exact mt19937 stream for a
    // tile of triangle pairs while the GPU evaluates the previous tile.
    constexpr size_t TARGET_TILE_BYTES = 32ull * 1024ull * 1024ull;
    size_t rowsPerTile = TARGET_TILE_BYTES /
        (n * static_cast<size_t>(RANDOMS_PER_PAIR) * sizeof(val_t));
    if (rowsPerTile < 1) rowsPerTile = 1;
    if (rowsPerTile > n) rowsPerTile = n;
    if (rowsPerTile > static_cast<size_t>(INT_MAX)) rowsPerTile = INT_MAX;

    const int R = static_cast<int>(rowsPerTile);
    const size_t maxCells = static_cast<size_t>(R) * n;
    const size_t maxRndBytes = maxCells * RANDOMS_PER_PAIR * sizeof(val_t);
    const size_t maxMaskBytes = maxCells;

    val_t* hostRnd[2] = {nullptr, nullptr};
    unsigned char* hostMask[2] = {nullptr, nullptr};
    val_t* devRnd[2] = {nullptr, nullptr};
    unsigned char* devMask[2] = {nullptr, nullptr};
    cudaStream_t streams[2];
    cudaEvent_t copyDone[2];
    bool used[2] = {false, false};

    for (int s = 0; s < 2; ++s) {
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hostRnd[s]), maxRndBytes),
                  "malloc host rnd");
        cudaCheck(cudaMallocHost(reinterpret_cast<void**>(&hostMask[s]), maxMaskBytes),
                  "malloc host mask");
        cudaCheck(cudaMalloc(&devRnd[s], maxRndBytes), "malloc dev rnd");
        cudaCheck(cudaMalloc(&devMask[s], maxMaskBytes), "malloc dev mask");
        cudaCheck(cudaStreamCreate(&streams[s]), "create stream");
        cudaCheck(cudaEventCreateWithFlags(&copyDone[s], cudaEventDisableTiming),
                  "create event");
    }

    const int block = 256;
    size_t tileIndex = 0;

    for (size_t base = 0; base < n; base += static_cast<size_t>(R), ++tileIndex) {
        const int rows = static_cast<int>(std::min<size_t>(static_cast<size_t>(R), n - base));
        const size_t cells = static_cast<size_t>(rows) * n;
        const size_t rndBytes = cells * RANDOMS_PER_PAIR * sizeof(val_t);
        const size_t maskBytes = cells;
        const int s = static_cast<int>(tileIndex & 1ull);

        if (used[s]) {
            cudaCheck(cudaEventSynchronize(copyDone[s]), "wait copy");
        }
        used[s] = true;

        val_t* rnd = hostRnd[s];
        unsigned char* mask = hostMask[s];

        for (int r = 0; r < rows; ++r) {
            const size_t i = base + static_cast<size_t>(r);
            const Triangle& triI = state.triangles[i];

            for (size_t j = 0; j < n; ++j) {
                const size_t cell = static_cast<size_t>(r) * n + j;
                mask[cell] = 0;

                if (i == j) continue;
                if (triI.normal().dot(state.triangles[j].normal()) > 0.99f) continue;

                mask[cell] = 1;
                val_t* dst = rnd + cell * RANDOMS_PER_PAIR;
                for (int k = 0; k < RANDOMS_PER_PAIR; ++k) {
                    dst[k] = rng.rand();
                }
            }

            if ((i + 1) % 100 == 0 || i + 1 == n) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, n);
            }
        }

        cudaCheck(cudaMemcpyAsync(devRnd[s], hostRnd[s], rndBytes,
                                  cudaMemcpyHostToDevice, streams[s]), "copy rnd");
        cudaCheck(cudaMemcpyAsync(devMask[s], hostMask[s], maskBytes,
                                  cudaMemcpyHostToDevice, streams[s]), "copy mask");
        cudaCheck(cudaEventRecord(copyDone[s], streams[s]), "record copy");

        const int total = rows * static_cast<int>(n);
        const int grid = (total + block - 1) / block;
        kijKernel<<<grid, block, 0, streams[s]>>>(
            state.dev.triangles, state.dev.nodes, state.dev.triIndices,
            state.dev.root, devMask[s], devRnd[s], state.dev.kij,
            static_cast<int>(n), static_cast<int>(base), rows);
        cudaCheck(cudaGetLastError(), "launch kij");
    }

    for (int s = 0; s < 2; ++s) {
        cudaCheck(cudaStreamSynchronize(streams[s]), "sync kij");
        cudaFree(devRnd[s]);
        cudaFree(devMask[s]);
        cudaFreeHost(hostRnd[s]);
        cudaFreeHost(hostMask[s]);
        cudaEventDestroy(copyDone[s]);
        cudaStreamDestroy(streams[s]);
    }

    if (g_validateResults) {
        cudaCheck(cudaMemcpy(state.kij.data(), state.dev.kij,
                             n * n * sizeof(val_t), cudaMemcpyDeviceToHost),
                  "copy kij back");
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int block = 256;
    const int grid = (n + block - 1) / block;

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simulationKernel<<<grid, block>>>(
            static_cast<int>(t), n, state.dev.kij, state.dev.tau,
            state.dev.radB, state.dev.radE, state.dev.rho, state.dev.areas,
            state.dev.radB);
        cudaCheck(cudaGetLastError(), "launch simulation");

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }

    cudaCheck(cudaDeviceSynchronize(), "simulation");

    if (g_validateResults) {
        cudaCheck(cudaMemcpy(state.radB.data(), state.dev.radB,
                             state.numTimesteps * state.numTriangles * sizeof(val_t),
                             cudaMemcpyDeviceToHost), "copy radB back");
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const int n = static_cast<int>(state.numTriangles);
    const int block = 256;
    const int grid = (n + block - 1) / block;

    distanceKernel<<<grid, block>>>(
        state.dev.radB, state.dev.distances, n,
        static_cast<int>(state.numTimesteps), static_cast<int>(state.sourceIndex));
    cudaCheck(cudaGetLastError(), "launch distances");
    cudaCheck(cudaDeviceSynchronize(), "distances");

    cudaCheck(cudaMemcpy(state.distances.data(), state.dev.distances,
                         state.numTriangles * sizeof(val_t), cudaMemcpyDeviceToHost),
              "copy distances back");
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

    g_validateResults = validate;

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
