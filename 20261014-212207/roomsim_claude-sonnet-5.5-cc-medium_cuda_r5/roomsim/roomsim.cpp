/**
 * Room Response Simulation Benchmark
 * 
 * This is a simplified implementation (CUDA-parallelized) of room impulse response
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
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
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
    HD val_t norm() const { return std::sqrt(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
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

    HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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
// Device-side Octree (flattened) and Triangle Data
// ============================================================================

struct DevOctNode {
    Vec3 center, halfExtent;
    int child[8];
    int triStart, triCount;
};

struct DevLeafTri {
    Vec3 a, b, c;
    uint32_t idx;
};

static int flattenOctree(const Octree& node, std::vector<DevOctNode>& nodes,
                         std::vector<DevLeafTri>& leafTris,
                         const std::vector<Triangle>& triangles) {
    int id = static_cast<int>(nodes.size());
    nodes.emplace_back();

    DevOctNode d;
    d.center = node.center;
    d.halfExtent = node.halfExtent;
    d.triStart = static_cast<int>(leafTris.size());
    d.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) {
        const Triangle& t = triangles[idx];
        leafTris.push_back({t.a, t.b, t.c, static_cast<uint32_t>(idx)});
    }
    for (int i = 0; i < 8; ++i) {
        d.child[i] = node.children[i]
                         ? flattenOctree(*node.children[i], nodes, leafTris, triangles)
                         : -1;
    }
    nodes[id] = d;
    return id;
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

    if (std::abs(det) < EPSILON) return FLT_MAX;

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
// Device Helpers: random numbers, visibility, form factors
// ============================================================================

constexpr int RNG_PER_PAIR = 4 * NUM_RAYS;           // random numbers consumed per active pair
constexpr uint32_t PAIR_SKIPPED = 0xFFFFFFFFu;       // marker for culled / diagonal pairs
constexpr int MAX_STACK = 96;

// Tempering + std::uniform_real_distribution<float>(0,1) conversion (libstdc++ semantics)
__device__ __forceinline__ val_t rawToUnit(uint32_t y) {
    y ^= y >> 11;
    y ^= (y << 7) & 0x9d2c5680u;
    y ^= (y << 15) & 0xefc60000u;
    y ^= y >> 18;
    val_t r = __uint2float_rn(y) * 0x1p-32f;
    return r >= 1.0f ? 0x1.fffffep-1f : r;
}

__device__ __forceinline__ Vec3 pointInTriangle(const Triangle& t, val_t u, val_t v) {
    if (u + v > 1.0f) {
        u = 1.0f - u;
        v = 1.0f - v;
    }
    Vec3 ab = t.b - t.a;
    Vec3 ac = t.c - t.a;
    return t.a + ab * u + ac * v;
}

__device__ __forceinline__ bool rayIntersectsBox(const DevOctNode& n, const Vec3& p1, const Vec3& p2) {
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

__device__ bool isRayBlocked(const Vec3& from, const Vec3& to,
                             const DevOctNode* nodes, const DevLeafTri* leafTris,
                             uint32_t srcTriIdx, uint32_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[MAX_STACK];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const DevOctNode& node = nodes[stack[--sp]];
        if (node.triCount > 0) {
            for (int k = 0; k < node.triCount; ++k) {
                const DevLeafTri& tri = leafTris[node.triStart + k];
                if (tri.idx == srcTriIdx || tri.idx == dstTriIdx) continue;
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 7; c >= 0; --c) {
                int ch = node.child[c];
                if (ch >= 0 && rayIntersectsBox(nodes[ch], from, to)) stack[sp++] = ch;
            }
        }
    }
    return false;
}

HD inline val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    val_t c = v.dot(normal) / vNorm;
    return ZERO < c ? c : ZERO;
}

// Pairs facing the same direction (and the diagonal) are skipped and consume no random numbers
__device__ __forceinline__ bool pairActive(const Triangle* tris, uint32_t i, uint32_t j) {
    if (i == j) return false;
    return !(tris[i].normal().dot(tris[j].normal()) > 0.99f);
}

// ============================================================================
// Kernels
// ============================================================================

// One block per row: counts active pairs per row, writes exclusive in-row pair
// offsets into the (uint32-aliased) Kij matrix.
constexpr int CULL_BLOCK = 256;
__global__ void __launch_bounds__(CULL_BLOCK)
cullOffsetKernel(const Triangle* tris, uint32_t N, uint32_t* out, uint32_t* rowCount) {
    __shared__ uint32_t scan[CULL_BLOCK];
    uint32_t i = blockIdx.x;
    uint32_t chunk = (N + blockDim.x - 1) / blockDim.x;
    uint32_t begin = min(N, threadIdx.x * chunk);
    uint32_t end = min(N, begin + chunk);

    uint32_t cnt = 0;
    for (uint32_t j = begin; j < end; ++j) cnt += pairActive(tris, i, j) ? 1u : 0u;

    scan[threadIdx.x] = cnt;
    __syncthreads();
    for (uint32_t s = 1; s < blockDim.x; s <<= 1) {
        uint32_t v = threadIdx.x >= s ? scan[threadIdx.x - s] : 0u;
        __syncthreads();
        scan[threadIdx.x] += v;
        __syncthreads();
    }
    uint32_t running = scan[threadIdx.x] - cnt;
    if (threadIdx.x == blockDim.x - 1) rowCount[i] = scan[threadIdx.x];

    uint32_t* row = out + static_cast<size_t>(i) * N;
    for (uint32_t j = begin; j < end; ++j) {
        if (pairActive(tris, i, j)) row[j] = running++;
        else row[j] = PAIR_SKIPPED;
    }
}

__global__ void tauKernel(const Triangle* tris, uint32_t N, int* tau) {
    uint32_t i = blockIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;
    int v = 0;
    if (i != j) {
        val_t dist = (tris[i].center() - tris[j].center()).norm();
        v = static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
    }
    tau[static_cast<size_t>(i) * N + j] = v;
}

constexpr int KIJ_BLOCK = 128;
__global__ void __launch_bounds__(KIJ_BLOCK)
formFactorKernel(const Triangle* tris, uint32_t N, uint32_t i0, const size_t* rowBase, size_t base0,
                 const uint32_t* stream, const DevOctNode* nodes, const DevLeafTri* leafTris,
                 val_t* kij) {
    uint32_t i = i0 + blockIdx.y;
    uint32_t j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= N) return;

    size_t pos = static_cast<size_t>(i) * N + j;
    uint32_t off = reinterpret_cast<uint32_t*>(kij)[pos];
    if (off == PAIR_SKIPPED) {
        kij[pos] = ZERO;
        return;
    }

    const Triangle triI = tris[i];
    const Triangle triJ = tris[j];
    const uint4* rnd = reinterpret_cast<const uint4*>(
        stream + (rowBase[i] - base0) + static_cast<size_t>(off) * RNG_PER_PAIR);

    val_t sum = ZERO;
    for (int r = 0; r < NUM_RAYS; ++r) {
        uint4 q = rnd[r];
        Vec3 pI = pointInTriangle(triI, rawToUnit(q.x), rawToUnit(q.y));
        Vec3 pJ = pointInTriangle(triJ, rawToUnit(q.z), rawToUnit(q.w));

        if (isRayBlocked(pI, pJ, nodes, leafTris, i, j)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        sum += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }
    kij[pos] = sum * INV_NUM_RAYS;
}

// One thread per receiving triangle i; j is accumulated in the original order
// so the floating-point result matches the sequential implementation.
__global__ void simStepKernel(uint32_t t, uint32_t N, const int* __restrict__ tau,
                              const val_t* __restrict__ kij, const val_t* __restrict__ areas,
                              const val_t* __restrict__ rho, const val_t* __restrict__ radE,
                              val_t* radB) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const int* tauRow = tau + static_cast<size_t>(i) * N;
    const val_t* kijRow = kij + static_cast<size_t>(i) * N;
    val_t sumB = ZERO;

    for (uint32_t j = 0; j < N; ++j) {
        if (i == j) continue;
        int tauij = tauRow[j];
        if (static_cast<int>(t) < tauij) continue;
        val_t k = kijRow[j];
        if (k <= ZERO) continue;
        val_t radJ = radB[static_cast<size_t>(t - static_cast<uint32_t>(tauij)) * N + j];
        if (radJ <= ZERO) continue;
        val_t w = k * areas[j];
        w = ONE < w ? ONE : w;
        sumB += w * radJ;
    }
    size_t o = static_cast<size_t>(t) * N + i;
    radB[o] = rho[i] * sumB + radE[o];
}

__global__ void transposeKernel(const val_t* in, val_t* out, uint32_t T, uint32_t N) {
    size_t idx = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<size_t>(T) * N) return;
    uint32_t t = static_cast<uint32_t>(idx / N);
    uint32_t n = static_cast<uint32_t>(idx % N);
    out[static_cast<size_t>(n) * T + t] = in[idx];
}

constexpr int DIST_BLOCK = 128;
__global__ void __launch_bounds__(DIST_BLOCK)
distanceKernel(const val_t* radBT, uint32_t T, uint32_t source, val_t* distances) {
    __shared__ val_t sVal[DIST_BLOCK];
    __shared__ int sIdx[DIST_BLOCK];
    const val_t* pBrow = radBT + static_cast<size_t>(blockIdx.x) * T;
    const val_t* pSrow = radBT + static_cast<size_t>(source) * T;

    val_t maxCorr = ZERO;
    int bestT = INT_MAX;
    for (uint32_t t = threadIdx.x; t < T; t += blockDim.x) {
        val_t sum = ZERO;
        for (uint32_t tt = t; tt < T; ++tt) sum += pSrow[tt - t] * pBrow[tt];
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }
    sVal[threadIdx.x] = maxCorr;
    sIdx[threadIdx.x] = bestT;
    __syncthreads();
    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            val_t ov = sVal[threadIdx.x + s];
            int oi = sIdx[threadIdx.x + s];
            if (ov > sVal[threadIdx.x] || (ov == sVal[threadIdx.x] && oi < sIdx[threadIdx.x])) {
                sVal[threadIdx.x] = ov;
                sIdx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        int b = (sVal[0] > ZERO) ? sIdx[0] : 0;
        distances[blockIdx.x] = WAVE_SPEED * static_cast<val_t>(b);
    }
}

__global__ void countNonZeroKernel(const val_t* kij, size_t count, unsigned long long* result) {
    unsigned long long local = 0;
    for (size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; i < count;
         i += static_cast<size_t>(gridDim.x) * blockDim.x) {
        if (kij[i] > EPSILON) local++;
    }
    for (int o = 16; o > 0; o >>= 1) local += __shfl_down_sync(0xffffffffu, local, o);
    if ((threadIdx.x & 31) == 0) atomicAdd(result, local);
}

// ============================================================================
// Host MT19937 state generator (raw twisted words; tempering happens on the GPU)
// ============================================================================

class RawMT19937 {
    static constexpr int N = 624, M = 397;
    uint32_t mt[N];
    int pos;

    void twist() {
        constexpr uint32_t UPPER = 0x80000000u, LOWER = 0x7fffffffu, MAG = 0x9908b0dfu;
        int k = 0;
        for (; k < N - M; ++k) {
            uint32_t y = (mt[k] & UPPER) | (mt[k + 1] & LOWER);
            mt[k] = mt[k + M] ^ (y >> 1) ^ ((y & 1u) ? MAG : 0u);
        }
        for (; k < N - 1; ++k) {
            uint32_t y = (mt[k] & UPPER) | (mt[k + 1] & LOWER);
            mt[k] = mt[k + (M - N)] ^ (y >> 1) ^ ((y & 1u) ? MAG : 0u);
        }
        uint32_t y = (mt[N - 1] & UPPER) | (mt[0] & LOWER);
        mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ ((y & 1u) ? MAG : 0u);
        pos = 0;
    }

public:
    explicit RawMT19937(uint32_t seed) : pos(N) {
        mt[0] = seed;
        for (int i = 1; i < N; ++i) mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + i;
    }

    void fill(uint32_t* out, size_t n) {
        while (n > 0) {
            if (pos == N) twist();
            size_t take = std::min(n, static_cast<size_t>(N - pos));
            memcpy(out, mt + pos, take * sizeof(uint32_t));
            pos += static_cast<int>(take);
            out += take;
            n -= take;
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
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> radE;        // Emission radiosity (T x N matrix)
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

    size_t sourceIndex;

    // Device data
    Triangle* d_tris = nullptr;
    DevOctNode* d_nodes = nullptr;
    DevLeafTri* d_leafTris = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    val_t* d_kij = nullptr;         // Form factors (N x N matrix, row-major)
    int* d_tau = nullptr;           // Time delays (N x N matrix, row-major)
    val_t* d_radE = nullptr;
    val_t* d_radB = nullptr;
    val_t* d_dist = nullptr;
    std::vector<uint32_t> rowCount; // Active pairs per row

    size_t idxTN(size_t t, size_t n) const { return t * numTriangles + n; }
};

template <typename T>
static T* deviceAlloc(size_t count) {
    T* p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, count * sizeof(T)));
    return p;
}

template <typename T>
static T* deviceUpload(const std::vector<T>& v) {
    T* p = deviceAlloc<T>(v.size());
    CUDA_CHECK(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
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

    // Initialize matrices
    state.radE.resize(timesteps * state.numTriangles, ZERO);
    state.radB.resize(timesteps * state.numTriangles, ZERO);
    state.distances.resize(state.numTriangles, ZERO);

    // Set source emission (active for first half of timesteps)
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        state.radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // Upload to device
    std::vector<DevOctNode> nodes;
    std::vector<DevLeafTri> leafTris;
    if (!state.triangles.empty()) flattenOctree(state.octree, nodes, leafTris, state.triangles);
    if (leafTris.empty()) leafTris.resize(1);

    size_t n = state.numTriangles;
    state.d_tris = deviceUpload(state.triangles);
    state.d_nodes = deviceUpload(nodes);
    state.d_leafTris = deviceUpload(leafTris);
    state.d_areas = deviceUpload(state.areas);
    state.d_rho = deviceUpload(state.rho);
    state.d_radE = deviceUpload(state.radE);
    state.d_radB = deviceAlloc<val_t>(std::max<size_t>(1, timesteps * n));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, timesteps * n * sizeof(val_t)));
    state.d_dist = deviceAlloc<val_t>(n);
    state.d_kij = deviceAlloc<val_t>(n * n);
    state.d_tau = deviceAlloc<int>(n * n);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const size_t N = state.numTriangles;
    const uint32_t N32 = static_cast<uint32_t>(N);

    // Pass 1: per-pair offsets into the sequential random stream (original consumption order)
    uint32_t* d_rowCount = deviceAlloc<uint32_t>(N);
    cullOffsetKernel<<<N32, CULL_BLOCK>>>(state.d_tris, N32, reinterpret_cast<uint32_t*>(state.d_kij),
                                          d_rowCount);
    CUDA_CHECK(cudaGetLastError());
    state.rowCount.resize(N);
    CUDA_CHECK(cudaMemcpy(state.rowCount.data(), d_rowCount, N * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rowCount));

    std::vector<size_t> rowBase(N + 1, 0);
    size_t maxRow = 0;
    for (size_t i = 0; i < N; ++i) {
        size_t nums = static_cast<size_t>(state.rowCount[i]) * RNG_PER_PAIR;
        rowBase[i + 1] = rowBase[i] + nums;
        maxRow = std::max(maxRow, nums);
    }
    const size_t total = rowBase[N];
    size_t* d_rowBase = deviceUpload(rowBase);

    // Pass 2: stream the random numbers to the GPU in row batches (host generates, GPU consumes)
    const size_t batchTarget = std::max(maxRow, std::min<size_t>(size_t(1) << 25,
                                                                  std::max<size_t>(total / 8, size_t(1) << 20)));
    uint32_t* h_buf[2];
    uint32_t* d_buf[2];
    cudaEvent_t copied[2], done[2];
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(&h_buf[b], std::max<size_t>(batchTarget, 4) * sizeof(uint32_t)));
        d_buf[b] = deviceAlloc<uint32_t>(std::max<size_t>(batchTarget, 4));
        CUDA_CHECK(cudaEventCreateWithFlags(&copied[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&done[b], cudaEventDisableTiming));
    }
    cudaStream_t sCopy, sComp;
    CUDA_CHECK(cudaStreamCreateWithFlags(&sCopy, cudaStreamNonBlocking));
    CUDA_CHECK(cudaStreamCreateWithFlags(&sComp, cudaStreamNonBlocking));

    RawMT19937 gen(42);
    size_t b0 = 0;
    int k = 0;
    while (b0 < N) {
        size_t b1 = b0 + 1;
        while (b1 < N && b1 - b0 < 65535 && rowBase[b1 + 1] - rowBase[b0] <= batchTarget) ++b1;
        const size_t nums = rowBase[b1] - rowBase[b0];
        const int buf = k & 1;

        if (k >= 2) {
            CUDA_CHECK(cudaEventSynchronize(copied[buf]));          // pinned buffer free again
            CUDA_CHECK(cudaStreamWaitEvent(sCopy, done[buf], 0));   // device buffer free again
        }
        gen.fill(h_buf[buf], nums);
        if (nums > 0) {
            CUDA_CHECK(cudaMemcpyAsync(d_buf[buf], h_buf[buf], nums * sizeof(uint32_t),
                                       cudaMemcpyHostToDevice, sCopy));
        }
        CUDA_CHECK(cudaEventRecord(copied[buf], sCopy));
        CUDA_CHECK(cudaStreamWaitEvent(sComp, copied[buf], 0));

        dim3 grid((N32 + KIJ_BLOCK - 1) / KIJ_BLOCK, static_cast<unsigned>(b1 - b0));
        formFactorKernel<<<grid, KIJ_BLOCK, 0, sComp>>>(
            state.d_tris, N32, static_cast<uint32_t>(b0), d_rowBase, rowBase[b0], d_buf[buf],
            state.d_nodes, state.d_leafTris, state.d_kij);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(done[buf], sComp));

        for (size_t i = b0; i < b1; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == N) {
                printf("  Progress: %zu/%zu triangles\n", i + 1, N);
            }
        }
        b0 = b1;
        ++k;
    }

    CUDA_CHECK(cudaStreamSynchronize(sComp));
    CUDA_CHECK(cudaStreamSynchronize(sCopy));
    CUDA_CHECK(cudaStreamDestroy(sCopy));
    CUDA_CHECK(cudaStreamDestroy(sComp));
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaFreeHost(h_buf[b]));
        CUDA_CHECK(cudaFree(d_buf[b]));
        CUDA_CHECK(cudaEventDestroy(copied[b]));
        CUDA_CHECK(cudaEventDestroy(done[b]));
    }
    CUDA_CHECK(cudaFree(d_rowBase));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const uint32_t N32 = static_cast<uint32_t>(state.numTriangles);
    dim3 grid((N32 + 255) / 256, N32);
    tauKernel<<<grid, 256>>>(state.d_tris, N32, state.d_tau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const uint32_t N32 = static_cast<uint32_t>(state.numTriangles);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        simStepKernel<<<(N32 + 63) / 64, 64>>>(static_cast<uint32_t>(t), N32, state.d_tau, state.d_kij,
                                               state.d_areas, state.d_rho, state.d_radE, state.d_radB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    val_t* d_radBT = deviceAlloc<val_t>(std::max<size_t>(1, T * N));
    size_t count = T * N;
    if (count > 0) {
        transposeKernel<<<static_cast<unsigned>((count + 255) / 256), 256>>>(
            state.d_radB, d_radBT, static_cast<uint32_t>(T), static_cast<uint32_t>(N));
        CUDA_CHECK(cudaGetLastError());
    }
    distanceKernel<<<static_cast<unsigned>(N), DIST_BLOCK>>>(
        d_radBT, static_cast<uint32_t>(T), static_cast<uint32_t>(state.sourceIndex), state.d_dist);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_dist, N * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB, T * N * sizeof(val_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_radBT));
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
    size_t total = state.numTriangles * state.numTriangles;
    unsigned long long* d_count = deviceAlloc<unsigned long long>(1);
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    countNonZeroKernel<<<1024, 256>>>(state.d_kij, total, d_count);
    CUDA_CHECK(cudaGetLastError());
    unsigned long long h_count = 0;
    CUDA_CHECK(cudaMemcpy(&h_count, d_count, sizeof(h_count), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));
    int nonZeroKij = static_cast<int>(h_count);
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, total,
           100.0f * nonZeroKij / static_cast<val_t>(total));

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

    // Initialize CUDA context up front so driver start-up is not attributed to any phase
    CUDA_CHECK(cudaFree(nullptr));

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
    bool ok = true;
    if (validate) {
        ok = validateResults(state);
    }

    CUDA_CHECK(cudaFree(state.d_tris));
    CUDA_CHECK(cudaFree(state.d_nodes));
    CUDA_CHECK(cudaFree(state.d_leafTris));
    CUDA_CHECK(cudaFree(state.d_areas));
    CUDA_CHECK(cudaFree(state.d_rho));
    CUDA_CHECK(cudaFree(state.d_kij));
    CUDA_CHECK(cudaFree(state.d_tau));
    CUDA_CHECK(cudaFree(state.d_radE));
    CUDA_CHECK(cudaFree(state.d_radB));
    CUDA_CHECK(cudaFree(state.d_dist));

    return ok ? 0 : 1;
}
