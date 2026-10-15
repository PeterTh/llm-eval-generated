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

#include <mpi.h>
#include <omp.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cfloat>
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

using idx_t = uint32_t;
using val_t = float;

#define HD __host__ __device__

#define CUDA_CHECK(call)                                                          \
    do {                                                                          \
        cudaError_t err_ = (call);                                                \
        if (err_ != cudaSuccess) {                                                \
            fprintf(stderr, "CUDA error %s at %s:%d\n", cudaGetErrorString(err_), \
                    __FILE__, __LINE__);                                          \
            MPI_Abort(MPI_COMM_WORLD, 1);                                         \
        }                                                                         \
    } while (0)

static int g_rank = 0;
static int g_size = 1;
#define ROOT_PRINTF(...) do { if (g_rank == 0) printf(__VA_ARGS__); } while (0)

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

    HD bool operator==(const Vec3& o) const {
        return fabsf(x - o.x) < EPSILON && fabsf(y - o.y) < EPSILON && fabsf(z - o.z) < EPSILON;
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
// Random Number Generation
// ============================================================================

// Counter-based generator (splitmix64 stream addressed by position). Every
// (i, j) pair owns a disjoint block of the stream, so form factors can be
// computed independently on any rank/thread with a reproducible result.
struct RandomGenerator {
    uint64_t state;

    HD explicit RandomGenerator(uint64_t seed, uint64_t pairIndex)
        : state(seed * 0xD1B54A32D192ED03ULL + pairIndex * 64ULL * 0x9E3779B97F4A7C15ULL) {}

    HD val_t rand() {
        state += 0x9E3779B97F4A7C15ULL;
        uint64_t z = state;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        z ^= z >> 31;
        return static_cast<val_t>(z >> 40) * (1.0f / 16777216.0f);  // [0, 1)
    }
};

// Generate a random point inside a triangle using barycentric coordinates
HD Vec3 randomPointInTriangle(const Triangle& t, RandomGenerator& rng) {
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
// Flattened Octree (device-friendly) and Visibility Testing
// ============================================================================

struct GpuNode {
    Vec3 center, halfExtent;
    int child[8];
    int triStart, triCount;  // triCount > 0 -> leaf
};

struct GpuOctree {
    std::vector<GpuNode> nodes;
    std::vector<int> leafTris;
};

static int flattenOctree(const Octree& node, GpuOctree& out) {
    int self = static_cast<int>(out.nodes.size());
    out.nodes.emplace_back();
    GpuNode g;
    g.center = node.center;
    g.halfExtent = node.halfExtent;
    g.triStart = static_cast<int>(out.leafTris.size());
    g.triCount = static_cast<int>(node.triangleIndices.size());
    for (size_t idx : node.triangleIndices) out.leafTris.push_back(static_cast<int>(idx));
    for (int c = 0; c < 8; ++c) g.child[c] = -1;
    if (g.triCount == 0) {
        for (int c = 0; c < 8; ++c) {
            if (node.children[c]) g.child[c] = flattenOctree(*node.children[c], out);
        }
    }
    out.nodes[self] = g;
    return self;
}

struct DeviceScene {
    const Triangle* tris;
    const GpuNode* nodes;
    const int* leafTris;
};

__device__ __forceinline__ bool rayIntersectsBox(const GpuNode& n, const Vec3& p1, const Vec3& p2) {
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

// Check if a ray between two triangles is blocked by any other triangle
__device__ bool isRayBlocked(const Vec3& from, const Vec3& to, const DeviceScene& scene,
                             int srcTriIdx, int dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    int stack[96];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const GpuNode& n = scene.nodes[stack[--sp]];
        if (n.triCount > 0) {
            for (int k = 0; k < n.triCount; ++k) {
                int idx = scene.leafTris[n.triStart + k];
                if (idx == srcTriIdx || idx == dstTriIdx) continue;
                const Triangle& tri = scene.tris[idx];
                val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
                if (dist > EPSILON && dist < rayLen - EPSILON) return true;
            }
        } else {
            for (int c = 0; c < 8; ++c) {
                int ci = n.child[c];
                if (ci >= 0 && rayIntersectsBox(scene.nodes[ci], from, to)) stack[sp++] = ci;
            }
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
__device__ val_t computeKij(int idxI, int idxJ, const DeviceScene& scene) {
    const Triangle triI = scene.tris[idxI];
    const Triangle triJ = scene.tris[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    RandomGenerator rng(42, static_cast<uint64_t>(idxI) * 0x100000000ULL + static_cast<uint64_t>(idxJ));
    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, scene, idxI, idxJ)) continue;

        Vec3 v = pJ - pI;
        val_t distSqr = v.squaredNorm();
        if (distSqr < EPSILON) continue;

        val_t cosPhiI = cosPhi(v, triI.normal());
        val_t cosPhiJ = cosPhi(-v, triJ.normal());

        if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

        kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
    }

    return kij * INV_NUM_RAYS;
}

// Stores w = min(Kij * area_j, 1) (0 where Kij <= 0) for the locally owned rows.
__global__ void formFactorKernel(DeviceScene scene, const val_t* areas, val_t* w,
                                 unsigned long long* nonZeroCount,
                                 size_t n, size_t rowBegin, size_t rows) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    bool active = p < rows * n;
    bool nonZero = false;
    if (active) {
        size_t i = rowBegin + p / n;
        size_t j = p % n;
        val_t kij = ZERO;
        if (i != j) kij = computeKij(static_cast<int>(i), static_cast<int>(j), scene);
        w[p] = kij > ZERO ? fminf(kij * areas[j], ONE) : ZERO;
        nonZero = kij > EPSILON;
    }
    unsigned ballot = __ballot_sync(0xffffffffu, nonZero);
    if ((threadIdx.x & 31) == 0 && ballot) atomicAdd(nonZeroCount, static_cast<unsigned long long>(__popc(ballot)));
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

HD int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(ceilf(dist * INV_WAVE_SPEED));
}

__global__ void tauKernel(const Triangle* tris, uint8_t* tau, size_t n, size_t rowBegin, size_t rows) {
    size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (p >= rows * n) return;
    size_t i = rowBegin + p / n;
    size_t j = p % n;
    int t = 0;
    if (i != j) t = computeTau(tris[i], tris[j]);
    tau[p] = static_cast<uint8_t>(t > 255 ? 255 : t);
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity of the locally owned triangles
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix), filled on rank 0 only
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    GpuOctree gpuOctree;

    size_t sourceIndex;
    size_t timeOff;                 // Source emits for t < timeOff

    // Row partition (MPI): this rank owns triangles [rowBegin, rowBegin + rows)
    size_t rowBegin = 0, rows = 0;
    std::vector<int> counts, displs;

    // Device data
    Triangle* dTris = nullptr;
    GpuNode* dNodes = nullptr;
    int* dLeaf = nullptr;
    val_t* dAreas = nullptr;
    val_t* dRho = nullptr;          // local rows
    val_t* dW = nullptr;            // rows x N, min(Kij * area_j, 1)
    uint8_t* dTau = nullptr;        // rows x N
    val_t* dRadB = nullptr;         // T x N
    val_t* dDist = nullptr;         // local rows
    unsigned long long* dNonZero = nullptr;
    unsigned long long nonZeroKij = 0;  // global count
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
    const size_t n = state.numTriangles;

    ROOT_PRINTF("Generated icosphere mesh with %zu triangles\n", n);

    // Build octree for spatial acceleration
    ROOT_PRINTF("Building octree...\n");
    state.octree.build(state.triangles);
    flattenOctree(state.octree, state.gpuOctree);

    // Initialize areas
    state.areas.resize(n);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Row partition across ranks
    state.counts.resize(g_size);
    state.displs.resize(g_size);
    size_t acc = 0;
    for (int r = 0; r < g_size; ++r) {
        size_t cnt = n / g_size + (static_cast<size_t>(r) < n % g_size ? 1 : 0);
        state.counts[r] = static_cast<int>(cnt);
        state.displs[r] = static_cast<int>(acc);
        acc += cnt;
    }
    state.rowBegin = static_cast<size_t>(state.displs[g_rank]);
    state.rows = static_cast<size_t>(state.counts[g_rank]);

    // Initialize reflectivity (local rows)
    state.rho.assign(state.rows, reflectivity);
    state.distances.assign(n, ZERO);

    // Source emission is active for the first half of timesteps
    state.timeOff = timesteps / 2;

    // Device allocations
    size_t need = state.rows * n * (sizeof(val_t) + sizeof(uint8_t)) + timesteps * n * sizeof(val_t);
    size_t freeMem = 0, totalMem = 0;
    CUDA_CHECK(cudaMemGetInfo(&freeMem, &totalMem));
    if (need > freeMem) {
        fprintf(stderr, "Rank %d: problem needs %.2f GB of device memory but only %.2f GB free; use more ranks\n",
                g_rank, need / 1e9, freeMem / 1e9);
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaMalloc(&state.dTris, n * sizeof(Triangle)));
    CUDA_CHECK(cudaMalloc(&state.dNodes, state.gpuOctree.nodes.size() * sizeof(GpuNode)));
    CUDA_CHECK(cudaMalloc(&state.dLeaf, std::max<size_t>(1, state.gpuOctree.leafTris.size()) * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.dAreas, n * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dRho, std::max<size_t>(1, state.rows) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dW, std::max<size_t>(1, state.rows * n) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dTau, std::max<size_t>(1, state.rows * n)));
    CUDA_CHECK(cudaMalloc(&state.dRadB, std::max<size_t>(1, timesteps * n) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dDist, std::max<size_t>(1, state.rows) * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.dNonZero, sizeof(unsigned long long)));

    CUDA_CHECK(cudaMemcpy(state.dTris, state.triangles.data(), n * sizeof(Triangle), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dNodes, state.gpuOctree.nodes.data(),
                          state.gpuOctree.nodes.size() * sizeof(GpuNode), cudaMemcpyHostToDevice));
    if (!state.gpuOctree.leafTris.empty())
        CUDA_CHECK(cudaMemcpy(state.dLeaf, state.gpuOctree.leafTris.data(),
                              state.gpuOctree.leafTris.size() * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.dAreas, state.areas.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
    if (state.rows)
        CUDA_CHECK(cudaMemcpy(state.dRho, state.rho.data(), state.rows * sizeof(val_t), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(state.dRadB, 0, std::max<size_t>(1, timesteps * n) * sizeof(val_t)));
    CUDA_CHECK(cudaMemset(state.dNonZero, 0, sizeof(unsigned long long)));
}

void releaseDevice(SimulationState& state) {
    cudaFree(state.dTris);  cudaFree(state.dNodes); cudaFree(state.dLeaf);
    cudaFree(state.dAreas); cudaFree(state.dRho);   cudaFree(state.dW);
    cudaFree(state.dTau);   cudaFree(state.dRadB);  cudaFree(state.dDist);
    cudaFree(state.dNonZero);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    ROOT_PRINTF("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    const size_t total = state.rows * n;
    if (total) {
        const int block = 128;
        size_t grid = (total + block - 1) / block;
        DeviceScene scene{state.dTris, state.dNodes, state.dLeaf};
        formFactorKernel<<<static_cast<unsigned>(grid), block>>>(
            scene, state.dAreas, state.dW, state.dNonZero, n, state.rowBegin, state.rows);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    unsigned long long local = 0;
    CUDA_CHECK(cudaMemcpy(&local, state.dNonZero, sizeof(local), cudaMemcpyDeviceToHost));
    MPI_Allreduce(&local, &state.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, MPI_COMM_WORLD);
    ROOT_PRINTF("  Progress: %zu/%zu triangles\n", n, n);
}

void computeTimeDelays(SimulationState& state) {
    ROOT_PRINTF("Computing time delays (Tau)...\n");
    const size_t total = state.rows * state.numTriangles;
    if (total) {
        const int block = 256;
        size_t grid = (total + block - 1) / block;
        tauKernel<<<static_cast<unsigned>(grid), block>>>(
            state.dTris, state.dTau, state.numTriangles, state.rowBegin, state.rows);
        CUDA_CHECK(cudaGetLastError());
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// One warp per local row i; lanes stride over j.
__global__ void simStepKernel(int t, size_t n, size_t rowBegin, size_t rows,
                              const uint8_t* __restrict__ tau, const val_t* __restrict__ w,
                              val_t* radB, const val_t* __restrict__ rho,
                              size_t sourceIndex, size_t timeOff) {
    size_t warp = (static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x) >> 5;
    int lane = threadIdx.x & 31;
    if (warp >= rows) return;
    size_t i = rowBegin + warp;
    const uint8_t* tauRow = tau + warp * n;
    const val_t* wRow = w + warp * n;

    val_t sumB = ZERO;
    for (size_t j = lane; j < n; j += 32) {
        if (j == i) continue;
        int tauij = tauRow[j];
        if (t < tauij) continue;
        val_t wij = wRow[j];
        if (wij <= ZERO) continue;
        val_t radJ = radB[static_cast<size_t>(t - tauij) * n + j];
        if (radJ <= ZERO) continue;
        sumB += wij * radJ;
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) sumB += __shfl_xor_sync(0xffffffffu, sumB, off);

    if (lane == 0) {
        val_t emit = (i == sourceIndex && static_cast<size_t>(t) < timeOff) ? ONE : ZERO;
        radB[static_cast<size_t>(t) * n + i] = rho[warp] * sumB + emit;
    }
}

void runSimulation(SimulationState& state) {
    ROOT_PRINTF("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles;
    std::vector<val_t> hRow(n);

    for (size_t t = 0; t < state.numTimesteps; ++t) {
        if (state.rows) {
            const int block = 256;  // 8 warps = 8 rows per block
            size_t grid = (state.rows * 32 + block - 1) / block;
            simStepKernel<<<static_cast<unsigned>(grid), block>>>(
                static_cast<int>(t), n, state.rowBegin, state.rows, state.dTau, state.dW,
                state.dRadB, state.dRho, state.sourceIndex, state.timeOff);
            CUDA_CHECK(cudaGetLastError());
        }

        if (g_size > 1) {
            // Exchange this timestep's radiosity among all ranks
            if (state.rows)
                CUDA_CHECK(cudaMemcpy(hRow.data() + state.rowBegin, state.dRadB + t * n + state.rowBegin,
                                      state.rows * sizeof(val_t), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, hRow.data(), state.counts.data(),
                           state.displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
            CUDA_CHECK(cudaMemcpy(state.dRadB + t * n, hRow.data(), n * sizeof(val_t), cudaMemcpyHostToDevice));
        }

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            ROOT_PRINTF("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

// One block per local triangle; threads stride over the lag t. The inner sum
// keeps the original (ascending tt) order; ties resolve to the smallest lag.
template <bool kShared>
__global__ void distanceKernel(const val_t* radB, size_t n, size_t numT, size_t rowBegin,
                               size_t sourceIndex, val_t* dist) {
    extern __shared__ val_t series[];  // [pB(T) | pS(T)] when kShared
    __shared__ val_t redSum[128];
    __shared__ int redT[128];

    size_t i = rowBegin + blockIdx.x;
    if (kShared) {
        for (size_t tt = threadIdx.x; tt < numT; tt += blockDim.x) {
            series[tt] = radB[tt * n + i];
            series[numT + tt] = radB[tt * n + sourceIndex];
        }
        __syncthreads();
    }

    val_t maxCorr = ZERO;
    int bestT = 0;
    for (size_t t = threadIdx.x; t < numT; t += blockDim.x) {
        val_t sum = ZERO;
        for (size_t tt = t; tt < numT; ++tt) {
            val_t pB, pS;
            if (kShared) {
                pB = series[tt];
                pS = series[numT + tt - t];
            } else {
                pB = radB[tt * n + i];
                pS = radB[(tt - t) * n + sourceIndex];
            }
            sum += pS * pB;
        }
        if (sum > maxCorr) {
            maxCorr = sum;
            bestT = static_cast<int>(t);
        }
    }
    redSum[threadIdx.x] = maxCorr;
    redT[threadIdx.x] = bestT;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            val_t a = redSum[threadIdx.x], b = redSum[threadIdx.x + s];
            int ta = redT[threadIdx.x], tb = redT[threadIdx.x + s];
            if (b > a || (b == a && b > ZERO && tb < ta)) {
                redSum[threadIdx.x] = b;
                redT[threadIdx.x] = tb;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) dist[blockIdx.x] = WAVE_SPEED * static_cast<val_t>(redT[0]);
}

void computeDistances(SimulationState& state) {
    ROOT_PRINTF("Computing distances via cross-correlation...\n");
    const size_t n = state.numTriangles;
    const size_t numT = state.numTimesteps;

    if (state.rows) {
        size_t shBytes = 2 * numT * sizeof(val_t);
        if (shBytes <= 48 * 1024) {
            distanceKernel<true><<<static_cast<unsigned>(state.rows), 128, shBytes>>>(
                state.dRadB, n, numT, state.rowBegin, state.sourceIndex, state.dDist);
        } else {
            distanceKernel<false><<<static_cast<unsigned>(state.rows), 128, 0>>>(
                state.dRadB, n, numT, state.rowBegin, state.sourceIndex, state.dDist);
        }
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(state.distances.data() + state.rowBegin, state.dDist,
                              state.rows * sizeof(val_t), cudaMemcpyDeviceToHost));
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    if (g_size > 1) {
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, state.distances.data(), state.counts.data(),
                       state.displs.data(), MPI_FLOAT, MPI_COMM_WORLD);
    }
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
    #pragma omp parallel for schedule(static) reduction(+ : receivedEnergy)
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

    // Check Kij matrix (should have some non-zero entries; counted on the devices)
    unsigned long long nonZeroKij = state.nonZeroKij;
    const size_t totalKij = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
           nonZeroKij, totalKij,
           100.0f * static_cast<val_t>(nonZeroKij) / static_cast<val_t>(totalKij));

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

static int run(int argc, char** argv) {
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
            ROOT_PRINTF("Unknown option: %s\n", argv[i]);
            printUsage(argv[0]);
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    ROOT_PRINTF("Room Response Simulation Benchmark\n");
    ROOT_PRINTF("===================================\n");
    ROOT_PRINTF("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
    ROOT_PRINTF("Timesteps: %d\n", timesteps);
    ROOT_PRINTF("Source triangle: %d\n", sourceIdx);
    ROOT_PRINTF("Reflectivity: %.2f\n", reflectivity);
    ROOT_PRINTF("Validation: %s\n", validate ? "enabled" : "disabled");
    ROOT_PRINTF("\n");

    // Initialize
    SimulationState state;
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);

    ROOT_PRINTF("\n");

    // Precomputation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = std::chrono::high_resolution_clock::now();

    computeTimeDelays(state);
    computeFormFactors(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endPre = std::chrono::high_resolution_clock::now();
    auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();

    ROOT_PRINTF("Precomputation time: %ld ms\n", preDuration);
    ROOT_PRINTF("\n");

    // Simulation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startSim = std::chrono::high_resolution_clock::now();

    runSimulation(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endSim = std::chrono::high_resolution_clock::now();
    auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();

    ROOT_PRINTF("Simulation time: %ld ms\n", simDuration);
    ROOT_PRINTF("\n");

    // Distance computation
    MPI_Barrier(MPI_COMM_WORLD);
    auto startDist = std::chrono::high_resolution_clock::now();

    computeDistances(state);

    MPI_Barrier(MPI_COMM_WORLD);
    auto endDist = std::chrono::high_resolution_clock::now();
    auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    ROOT_PRINTF("Distance computation time: %ld ms\n", distDuration);
    ROOT_PRINTF("\n");

    // Total time
    long totalTime = preDuration + simDuration + distDuration;
    ROOT_PRINTF("Total computation time: %ld ms\n", totalTime);

    // Performance metrics
    size_t n = state.numTriangles;
    size_t t = state.numTimesteps;
    double kijOps = static_cast<double>(n * n);
    double simOps = static_cast<double>(n * n * t);
    double distOps = static_cast<double>(n * t * t);

    ROOT_PRINTF("\nPerformance:\n");
    ROOT_PRINTF("  Triangles: %zu\n", n);
    ROOT_PRINTF("  Timesteps: %zu\n", t);
    ROOT_PRINTF("  Form factor computations: %.2e\n", kijOps);
    ROOT_PRINTF("  Simulation operations: %.2e\n", simOps);
    ROOT_PRINTF("  Distance computations: %.2e\n", distOps);
    ROOT_PRINTF("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

    // Memory usage
    size_t memKij = n * n * sizeof(val_t);
    size_t memTau = n * n * sizeof(int);
    size_t memRad = 2 * t * n * sizeof(val_t);
    size_t totalMem = memKij + memTau + memRad;
    ROOT_PRINTF("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

    // Hash
    uint64_t hash = computeHash(state);
    ROOT_PRINTF("  Result hash: %016lX\n", hash);
    ROOT_PRINTF("\n");
    
    // Print results for external validation
    if (printResults && g_rank == 0) {
        // Convert distances to double for output
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    // Validation
    int ok = 1;
    if (validate) {
        if (g_rank == 0) {
            state.radB.resize(state.numTimesteps * state.numTriangles);
            CUDA_CHECK(cudaMemcpy(state.radB.data(), state.dRadB, state.radB.size() * sizeof(val_t),
                                  cudaMemcpyDeviceToHost));
            ok = validateResults(state) ? 1 : 0;
        }
        MPI_Bcast(&ok, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    releaseDevice(state);
    return ok ? 0 : 1;
}

int main(int argc, char** argv) {
    int provided = 0;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &g_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &g_size);

    // Bind each rank to a GPU of its node
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, g_rank, MPI_INFO_NULL, &local);
    int localRank = 0, localSize = 1;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_size(local, &localSize);
    MPI_Comm_free(&local);
    int numDevices = 0;
    CUDA_CHECK(cudaGetDeviceCount(&numDevices));
    if (numDevices == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    CUDA_CHECK(cudaSetDevice(localRank % numDevices));
    CUDA_CHECK(cudaFree(nullptr));

    int rc = run(argc, argv);
    MPI_Finalize();
    return rc;
}
