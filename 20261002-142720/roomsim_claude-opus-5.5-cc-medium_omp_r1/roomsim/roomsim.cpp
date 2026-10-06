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
#include <atomic>
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

#include <immintrin.h>
#include <omp.h>
#include <sched.h>

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

// Flattened, cache-friendly copy of the octree used for read-only ray queries.
// Traversal visits exactly the same nodes/triangles as Octree::applyToTris and
// leaf triangles carry their precomputed edge vectors (bit-identical to the
// values computed on the fly by the ray-triangle test).
struct FlatOctree {
    struct Node {
        Vec3 center, halfExtent;
        uint32_t first;   // leaf: first entry in leafTris; inner: first child node
        uint32_t count;   // leaf: number of triangles; inner: number of children
        bool leaf;
    };
    // Leaf triangles in structure-of-arrays layout (vertex a, edges b-a and c-a)
    struct LeafTris {
        std::vector<val_t> v0x, v0y, v0z, e1x, e1y, e1z, e2x, e2y, e2z;
        std::vector<uint32_t> idx;
        size_t size() const { return idx.size(); }
        void clear() {
            for (auto* v : {&v0x, &v0y, &v0z, &e1x, &e1y, &e1z, &e2x, &e2y, &e2z}) v->clear();
            idx.clear();
        }
        void push(const Vec3& v0, const Vec3& e1, const Vec3& e2, uint32_t i) {
            v0x.push_back(v0.x); v0y.push_back(v0.y); v0z.push_back(v0.z);
            e1x.push_back(e1.x); e1y.push_back(e1.y); e1z.push_back(e1.z);
            e2x.push_back(e2.x); e2y.push_back(e2.y); e2z.push_back(e2.z);
            idx.push_back(i);
        }
    };
    std::vector<Node> nodes;
    LeafTris leafTris;

    void build(const Octree& root, const std::vector<Triangle>& tris) {
        nodes.clear();
        leafTris.clear();
        nodes.push_back({});
        fill(0, root, tris);
    }

private:
    void fill(size_t slot, const Octree& o, const std::vector<Triangle>& tris) {
        Node node;
        node.center = o.center;
        node.halfExtent = o.halfExtent;
        node.leaf = !o.triangleIndices.empty();
        if (node.leaf) {
            node.first = static_cast<uint32_t>(leafTris.size());
            node.count = static_cast<uint32_t>(o.triangleIndices.size());
            for (size_t idx : o.triangleIndices) {
                const Triangle& t = tris[idx];
                leafTris.push(t.a, t.b - t.a, t.c - t.a, static_cast<uint32_t>(idx));
            }
            nodes[slot] = node;
            return;
        }
        // Children are stored contiguously in their original order
        node.first = static_cast<uint32_t>(nodes.size());
        node.count = 0;
        for (int i = 0; i < 8; ++i) if (o.children[i]) ++node.count;
        nodes.resize(nodes.size() + node.count);
        nodes[slot] = node;
        uint32_t k = node.first;
        for (int i = 0; i < 8; ++i) {
            if (o.children[i]) fill(k++, *o.children[i], tris);
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
// Random Number Generation
// ============================================================================

// Bit-exact reimplementation of std::mt19937 that supports cheap forward
// skipping (twist-only, no tempering), so that parallel workers can jump to
// the exact position in the random stream the sequential code would be at.
class MT19937 {
    static constexpr int N = 624;
    static constexpr int M = 397;
    static constexpr uint32_t MATRIX_A = 0x9908b0dfu;
    static constexpr uint32_t UPPER = 0x80000000u;
    static constexpr uint32_t LOWER = 0x7fffffffu;
    alignas(64) uint32_t mt[N];
    int idx;

    static uint32_t mix(uint32_t a, uint32_t b, uint32_t c) {
        uint32_t y = (a & UPPER) | (b & LOWER);
        return c ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
    }

    void twist() {
        for (int k = 0; k < N - M; ++k) mt[k] = mix(mt[k], mt[k + 1], mt[k + M]);
        for (int k = N - M; k < N - 1; ++k) mt[k] = mix(mt[k], mt[k + 1], mt[k + M - N]);
        mt[N - 1] = mix(mt[N - 1], mt[0], mt[M - 1]);
        idx = 0;
    }

public:
    using result_type = uint32_t;
    static constexpr result_type min() { return 0u; }
    static constexpr result_type max() { return 0xffffffffu; }

    explicit MT19937(uint32_t seed = 5489u) {
        mt[0] = seed;
        for (int i = 1; i < N; ++i)
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        idx = N;
    }

    result_type operator()() {
        if (idx >= N) twist();
        uint32_t z = mt[idx++];
        z ^= (z >> 11);
        z ^= (z << 7) & 0x9d2c5680u;
        z ^= (z << 15) & 0xefc60000u;
        z ^= (z >> 18);
        return z;
    }

    void discard(uint64_t n) {
        while (n > 0) {
            if (idx >= N) twist();
            uint64_t avail = static_cast<uint64_t>(N - idx);
            uint64_t step = n < avail ? n : avail;
            idx += static_cast<int>(step);
            n -= step;
        }
    }
};

class RandomGenerator {
    MT19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    // Each rand() consumes exactly one 32-bit engine output (float needs 24 bits)
    void skip(uint64_t n) { rng.discard(n); }
    void setEngine(const MT19937& e) { rng = e; }
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
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

[[maybe_unused]] val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Visibility Testing (Octree-accelerated)
// ============================================================================

// Check if a ray between two triangles is blocked by any other triangle
// Uses octree for O(log N) instead of O(N) search
bool isRayBlocked(const Vec3& from, const Vec3& to,
                  const FlatOctree& octree,
                  size_t srcTriIdx, size_t dstTriIdx) {
    Vec3 dir = to - from;
    val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    Vec3 dirNorm = dir / rayLen;

    // Segment/box test terms that depend only on the ray (see Octree::rayIntersectsBox)
    const Vec3 d = (to - from) * 0.5f;
    const Vec3 mid = from + d;
    const Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};
    const val_t limit = rayLen - EPSILON;

    const FlatOctree::Node* nodes = octree.nodes.data();
    const FlatOctree::LeafTris& lt = octree.leafTris;
    const val_t* __restrict v0x = lt.v0x.data();
    const val_t* __restrict v0y = lt.v0y.data();
    const val_t* __restrict v0z = lt.v0z.data();
    const val_t* __restrict e1x = lt.e1x.data();
    const val_t* __restrict e1y = lt.e1y.data();
    const val_t* __restrict e1z = lt.e1z.data();
    const val_t* __restrict e2x = lt.e2x.data();
    const val_t* __restrict e2y = lt.e2y.data();
    const val_t* __restrict e2z = lt.e2z.data();
    const uint32_t* __restrict tidx = lt.idx.data();
    const uint32_t src = static_cast<uint32_t>(srcTriIdx);
    const uint32_t dst = static_cast<uint32_t>(dstTriIdx);

    auto boxHit = [&](const FlatOctree::Node& nd) {
        const Vec3& he = nd.halfExtent;
        Vec3 c = mid - nd.center;
        if (std::abs(c.x) > he.x + ad.x) return false;
        if (std::abs(c.y) > he.y + ad.y) return false;
        if (std::abs(c.z) > he.z + ad.z) return false;

        if (std::abs(d.y * c.z - d.z * c.y) > he.y * ad.z + he.z * ad.y + EPSILON) return false;
        if (std::abs(d.z * c.x - d.x * c.z) > he.z * ad.x + he.x * ad.z + EPSILON) return false;
        if (std::abs(d.x * c.y - d.y * c.x) > he.x * ad.y + he.y * ad.x + EPSILON) return false;
        return true;
    };

    // Iterative depth-first traversal (root is always visited, children are box-tested)
    uint32_t stack[512];
    int sp = 0;
    stack[sp++] = 0;
    while (sp > 0) {
        const FlatOctree::Node& nd = nodes[stack[--sp]];
        if (nd.leaf) {
            // Branch-free Moller-Trumbore over the whole leaf; performs the same
            // arithmetic as rayTriangleIntersect, and a rejected test (which
            // returns FLT_MAX there) can never satisfy dist < rayLen - EPSILON.
            int blocked = 0;
            const uint32_t end = nd.first + nd.count;
            #pragma omp simd reduction(|:blocked)
            for (uint32_t k = nd.first; k < end; ++k) {
                const Vec3 e1 = {e1x[k], e1y[k], e1z[k]};
                const Vec3 e2 = {e2x[k], e2y[k], e2z[k]};
                const Vec3 v0 = {v0x[k], v0y[k], v0z[k]};
                Vec3 pvec = dirNorm.cross(e2);
                val_t det = e1.dot(pvec);
                val_t invDet = 1.0f / det;
                Vec3 tvec = from - v0;
                val_t u = tvec.dot(pvec) * invDet;
                Vec3 qvec = tvec.cross(e1);
                val_t v = dirNorm.dot(qvec) * invDet;
                val_t dist = e2.dot(qvec) * invDet;
                int hit = !(std::abs(det) < EPSILON) &
                          !(u < 0.0f) & !(u > 1.0f) &
                          !(v < 0.0f) & !(u + v > 1.0f) &
                          (dist > EPSILON) & (dist < limit) &
                          (tidx[k] != src) & (tidx[k] != dst);
                blocked |= hit;
            }
            if (blocked) return true;  // Ray is blocked
            continue;
        }
        // Push in reverse so children are visited in original order
        for (uint32_t k = nd.count; k-- > 0;) {
            uint32_t ci = nd.first + k;
            if (boxHit(nodes[ci])) stack[sp++] = ci;
        }
    }
    return false;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return std::max(ZERO, v.dot(normal) / vNorm);
}

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const FlatOctree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (triI.normal().dot(triJ.normal()) > 0.99f) return ZERO;

    val_t kij = ZERO;

    for (int r = 0; r < NUM_RAYS; ++r) {
        Vec3 pI = randomPointInTriangle(triI, rng);
        Vec3 pJ = randomPointInTriangle(triJ, rng);

        if (isRayBlocked(pI, pJ, octree, idxI, idxJ)) continue;

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

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

// Allocator that leaves elements uninitialized on resize, so large matrices can
// be first-touched in parallel (pages end up on the NUMA node that uses them).
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    template <typename U> struct rebind { using other = DefaultInitAllocator<U>; };
    DefaultInitAllocator() = default;
    template <typename U> DefaultInitAllocator(const DefaultInitAllocator<U>&) noexcept {}
    template <typename U> void construct(U* p) noexcept { ::new (static_cast<void*>(p)) U; }
    template <typename U, typename... Args> void construct(U* p, Args&&... args) {
        ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
    }
};

template <typename T>
using par_vector = std::vector<T, DefaultInitAllocator<T>>;

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    par_vector<val_t> kij;          // Form factors (N x N matrix, row-major)
    par_vector<int> tau;            // Time delays (N x N matrix, row-major)
    par_vector<val_t> radE;         // Emission radiosity (T x N matrix)
    par_vector<val_t> radB;         // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    FlatOctree flatOctree;          // Flattened copy used for ray queries

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
    state.flatOctree.build(state.octree, state.triangles);

    // Initialize areas
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices
    // Zero-fill in parallel, distributing rows the same way as the simulation loop
    const size_t n = state.numTriangles;
    state.kij.resize(n * n);
    state.tau.resize(n * n);
    state.radE.resize(timesteps * n);
    state.radB.resize(timesteps * n);
    #pragma omp parallel
    {
        #pragma omp for schedule(static) nowait
        for (size_t i = 0; i < n; ++i) {
            std::fill_n(&state.kij[i * n], n, ZERO);
            std::fill_n(&state.tau[i * n], n, 0);
        }
        #pragma omp for schedule(static) nowait
        for (size_t k = 0; k < timesteps * n; ++k) {
            state.radE[k] = ZERO;
            state.radB[k] = ZERO;
        }
    }
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
    printf("Computing form factors (Kij)...\n");
    const size_t n = state.numTriangles;
    const std::vector<Triangle>& tris = state.triangles;
    if (n == 0) return;

    // The sequential reference draws all form factors from a single RNG stream
    // (seed 42) in row-major order. Each non-culled off-diagonal pair consumes
    // exactly 4 * NUM_RAYS values; culled pairs and the diagonal consume none.
    // The work is split into (row, column block) items; for each item we compute
    // its offset in the stream so items can be processed in parallel while
    // every pair sees exactly the same random numbers as in the sequential code.
    constexpr uint64_t DRAWS_PER_PAIR = 4 * NUM_RAYS;
    const size_t nthreads = static_cast<size_t>(omp_get_max_threads());
    const size_t targetItems = std::max<size_t>(n, 32 * nthreads);
    const size_t blocksPerRow = std::min(n, (targetItems + n - 1) / n);
    const size_t colBlock = (n + blocksPerRow - 1) / blocksPerRow;
    const size_t numItems = n * blocksPerRow;

    // itemOffset[it] = stream position at the start of item `it`
    std::vector<uint64_t> itemOffset(numItems + 1, 0);
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        const Vec3 ni = tris[i].normal();
        for (size_t b = 0; b < blocksPerRow; ++b) {
            const size_t j0 = b * colBlock;
            const size_t j1 = std::min(n, j0 + colBlock);
            uint64_t c = 0;
            for (size_t j = j0; j < j1; ++j) {
                if (i == j) continue;
                if (ni.dot(tris[j].normal()) > 0.99f) continue;
                ++c;
            }
            itemOffset[i * blocksPerRow + b + 1] = c * DRAWS_PER_PAIR;
        }
    }
    for (size_t it = 0; it < numItems; ++it) itemOffset[it + 1] += itemOffset[it];

    // One thread sweeps the RNG stream once and publishes the engine state at the
    // start of every item; the other threads start computing immediately and
    // only wait if they get ahead of the sweep (which is far cheaper than Kij).
    std::vector<MT19937> snapshots(numItems);
    std::unique_ptr<std::atomic<uint32_t>[]> ready(new std::atomic<uint32_t>[numItems]);
    for (size_t it = 0; it < numItems; ++it) ready[it].store(0, std::memory_order_relaxed);
    std::unique_ptr<std::atomic<uint32_t>[]> rowBlocksLeft(new std::atomic<uint32_t>[n]);
    for (size_t i = 0; i < n; ++i)
        rowBlocksLeft[i].store(static_cast<uint32_t>(blocksPerRow), std::memory_order_relaxed);
    std::atomic<size_t> nextItem{0};
    std::atomic<size_t> rowsDone{0};

    #pragma omp parallel
    {
        if (omp_get_thread_num() == 0 || omp_get_num_threads() == 1) {
            MT19937 sweep(42);
            uint64_t pos = 0;
            for (size_t it = 0; it < numItems; ++it) {
                sweep.discard(itemOffset[it] - pos);
                pos = itemOffset[it];
                snapshots[it] = sweep;
                ready[it].store(1, std::memory_order_release);
            }
        }

        RandomGenerator rng(42);
        for (;;) {
            const size_t it = nextItem.fetch_add(1, std::memory_order_relaxed);
            if (it >= numItems) break;
            while (ready[it].load(std::memory_order_acquire) == 0) _mm_pause();
            rng.setEngine(snapshots[it]);

            const size_t i = it / blocksPerRow;
            const size_t j0 = (it % blocksPerRow) * colBlock;
            const size_t j1 = std::min(n, j0 + colBlock);
            val_t* row = &state.kij[state.idx2d(i, 0)];
            for (size_t j = j0; j < j1; ++j) {
                if (i == j) continue;
                row[j] = computeKij(i, j, tris, state.flatOctree, rng);
            }

            if (rowBlocksLeft[i].fetch_sub(1, std::memory_order_acq_rel) == 1) {
                const size_t done = rowsDone.fetch_add(1, std::memory_order_relaxed) + 1;
                if (done % 100 == 0 || done == n) {
                    #pragma omp critical(progress)
                    printf("  Progress: %zu/%zu triangles\n", done, n);
                }
            }
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        for (size_t j = 0; j < state.numTriangles; ++j) {
            if (i == j) continue;
            state.tau[state.idx2d(i, j)] = computeTau(
                state.triangles[i], state.triangles[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    // Every tau(i,j) >= 1 for distinct triangles, so timestep t only reads
    // radiosities of earlier timesteps: all triangles of a timestep are independent.
    #pragma omp parallel
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        #pragma omp for schedule(static)
        for (size_t i = 0; i < state.numTriangles; ++i) {
            val_t sumB = ZERO;

            for (size_t j = 0; j < state.numTriangles; ++j) {
                if (i == j) continue;

                int tauij = state.tau[state.idx2d(i, j)];

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                val_t kij = state.kij[state.idx2d(i, j)];
                if (kij <= ZERO) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = state.radB[state.idxTN(srcTime, j)];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(kij * state.areas[j], ONE) * radJ;
            }

            // Update radiosity: reflection + emission
            state.radB[state.idxTN(t, i)] = state.rho[i] * sumB + state.radE[state.idxTN(t, i)];
        }

        #pragma omp single nowait
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < state.numTimesteps; ++tt) {
                val_t pB = state.radB[state.idxTN(tt, i)];
                val_t pS = state.radB[state.idxTN(tt - t, state.sourceIndex)];
                sum += pS * pB;
            }

            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = static_cast<int>(t);
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
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
// Thread Placement
// ============================================================================

// Unless the user configured OpenMP thread affinity, pin each OpenMP thread to
// its own CPU: one hardware thread per physical core first (alternating
// between packages), SMT siblings last. Stable placement keeps the parallel
// first-touch NUMA layout valid and avoids migration stalls at barriers.
static int readSysfsInt(int cpu, const char* item) {
    char path[128];
    snprintf(path, sizeof(path), "/sys/devices/system/cpu/cpu%d/topology/%s", cpu, item);
    FILE* f = fopen(path, "r");
    if (!f) return -1;
    int v = -1;
    if (fscanf(f, "%d", &v) != 1) v = -1;
    fclose(f);
    return v;
}

void pinOpenMPThreads() {
    if (getenv("OMP_PROC_BIND") || getenv("OMP_PLACES") || getenv("GOMP_CPU_AFFINITY")) return;

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) return;

    struct CpuInfo { int cpu, smt, coreRank, pkg; };
    std::vector<CpuInfo> cpus;
    std::map<std::pair<int, int>, int> smtCount;     // (pkg, core) -> siblings seen
    std::map<std::pair<int, int>, int> coreRankOf;   // (pkg, core) -> rank in package
    std::map<int, int> coresInPkg;
    for (int c = 0; c < CPU_SETSIZE; ++c) {
        if (!CPU_ISSET(c, &allowed)) continue;
        int pkg = readSysfsInt(c, "physical_package_id");
        int core = readSysfsInt(c, "core_id");
        if (pkg < 0 || core < 0) { pkg = 0; core = c; }
        auto key = std::make_pair(pkg, core);
        if (!coreRankOf.count(key)) coreRankOf[key] = coresInPkg[pkg]++;
        cpus.push_back({c, smtCount[key]++, coreRankOf[key], pkg});
    }
    if (cpus.size() < 2) return;
    std::stable_sort(cpus.begin(), cpus.end(), [](const CpuInfo& a, const CpuInfo& b) {
        if (a.smt != b.smt) return a.smt < b.smt;
        if (a.coreRank != b.coreRank) return a.coreRank < b.coreRank;
        return a.pkg < b.pkg;
    });

    #pragma omp parallel
    {
        const size_t t = static_cast<size_t>(omp_get_thread_num());
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[t % cpus.size()].cpu, &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
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

    pinOpenMPThreads();

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
