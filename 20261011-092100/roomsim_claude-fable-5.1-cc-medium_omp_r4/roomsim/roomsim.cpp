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
#include <sstream>
#include <vector>

#include <omp.h>

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

// ============================================================================
// Flattened Octree (cache-friendly traversal; identical tests and visit set)
// ============================================================================

// Nodes are stored contiguously, all children of a node adjacent to each other,
// so that the ray/box tests of the children touch consecutive memory.
struct FlatOctreeNode {
    Vec3 center;
    Vec3 halfExtent;
    uint32_t firstChild;    // index of first child node
    uint32_t numChildren;   // 0 for leaves
    uint32_t triStart;      // leaf: offset into triIndices
    uint32_t triCount;      // leaf: number of triangles
};

// Per-ray constants shared by all box tests of one segment p1 -> p2
struct RayBoxQuery {
    Vec3 d;     // (p2 - p1) * 0.5
    Vec3 mid;   // p1 + d
    Vec3 ad;    // |d|
    RayBoxQuery(const Vec3& p1, const Vec3& p2)
        : d((p2 - p1) * 0.5f), mid(p1 + d), ad(std::abs(d.x), std::abs(d.y), std::abs(d.z)) {}
};

class FlatOctree {
public:
    std::vector<FlatOctreeNode> nodes;
    std::vector<uint32_t> triIndices;
    const std::vector<Triangle>* allTriangles = nullptr;

    void build(const Octree& root, const std::vector<Triangle>& triangles) {
        allTriangles = &triangles;
        nodes.clear();
        triIndices.clear();
        if (triangles.empty()) return;
        nodes.emplace_back();
        fill(root, 0);
    }

    // Same test as Octree::rayIntersectsBox, with the ray constants hoisted out
    static bool rayIntersectsBox(const FlatOctreeNode& nd, const RayBoxQuery& q) {
        const Vec3& d = q.d;
        const Vec3& ad = q.ad;
        const Vec3& halfExtent = nd.halfExtent;
        Vec3 c = q.mid - nd.center;

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
        if (nodes.empty()) return false;
        RayBoxQuery q(p1, p2);
        return visit(nodes[0], q, func);
    }

private:
    void fill(const Octree& src, size_t idx) {
        nodes[idx].center = src.center;
        nodes[idx].halfExtent = src.halfExtent;
        if (!src.triangleIndices.empty()) {
            nodes[idx].firstChild = 0;
            nodes[idx].numChildren = 0;
            nodes[idx].triStart = static_cast<uint32_t>(triIndices.size());
            nodes[idx].triCount = static_cast<uint32_t>(src.triangleIndices.size());
            for (size_t t : src.triangleIndices) triIndices.push_back(static_cast<uint32_t>(t));
            return;
        }
        // Allocate all children contiguously first, then fill them recursively
        uint32_t first = static_cast<uint32_t>(nodes.size());
        uint32_t count = 0;
        for (int i = 0; i < 8; ++i) if (src.children[i]) ++count;
        nodes.resize(nodes.size() + count);
        nodes[idx].firstChild = first;
        nodes[idx].numChildren = count;
        nodes[idx].triStart = 0;
        nodes[idx].triCount = 0;
        uint32_t k = 0;
        for (int i = 0; i < 8; ++i) {
            if (src.children[i]) {
                fill(*src.children[i], first + k);   // children in the original order 0..7
                ++k;
            }
        }
    }

    template<typename Func>
    bool visit(const FlatOctreeNode& nd, const RayBoxQuery& q, Func& func) const {
        if (nd.numChildren == 0) {
            const uint32_t* tris = triIndices.data() + nd.triStart;
            for (uint32_t k = 0; k < nd.triCount; ++k) {
                size_t idx = tris[k];
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }
        const FlatOctreeNode* child = nodes.data() + nd.firstChild;
        for (uint32_t k = 0; k < nd.numChildren; ++k) {
            if (rayIntersectsBox(child[k], q)) {
                if (visit(child[k], q, func)) return true;
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
    explicit RandomGenerator(const std::mt19937& engine) : rng(engine), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
};

// ============================================================================
// MT19937 Jump-Ahead
// ============================================================================
//
// The original (sequential) form-factor computation draws all random samples
// from a single std::mt19937 stream.  To compute form factors in parallel while
// reproducing exactly the same samples, every worker needs the generator state
// at an arbitrary offset into that stream.  mt19937::discard() is O(offset),
// which would cost as much as the actual work, so we jump ahead with the
// standard GF(2) polynomial method: the state transition A is linear, so
// A^v s = g(A) s with g(t) = t^v mod phi(t), where phi is the characteristic
// polynomial of A (degree 19937, obtained via Berlekamp-Massey).  g(A) s is
// then evaluated by Horner's rule using 19937 single steps of the recurrence.

namespace mtjump {

constexpr int MT_N = 624, MT_M = 397;
constexpr uint32_t MATRIX_A = 0x9908b0dfu, UPPER_MASK = 0x80000000u, LOWER_MASK = 0x7fffffffu;
constexpr int DEG = 19937;                 // degree of the characteristic polynomial
constexpr int PW = (DEG + 63) / 64;        // words for a polynomial of degree < DEG
constexpr int TW = PW + 2;                 // words for a shifted copy of a degree-DEG polynomial
constexpr int PRODW = 2 * PW + 2;          // words for an unreduced product

using Poly = std::vector<uint64_t>;        // bit i = coefficient of t^i

// Sliding-window state (x_n, ..., x_{n+623}): logical word k is x[(h + k) % MT_N]
struct State {
    uint32_t x[MT_N];
    int h = 0;
};

// Same initialisation as std::mt19937(seed)
inline void seedState(State& s, uint32_t seed) {
    s.x[0] = seed;
    for (int i = 1; i < MT_N; ++i) {
        uint32_t p = s.x[i - 1];
        s.x[i] = 1812433253u * (p ^ (p >> 30)) + static_cast<uint32_t>(i);
    }
    s.h = 0;
}

// One recurrence step: (x_n..x_{n+623}) -> (x_{n+1}..x_{n+624}); returns x_{n+624}
inline uint32_t step(State& s) {
    int h = s.h;
    int h1 = h + 1; if (h1 >= MT_N) h1 -= MT_N;
    int hm = h + MT_M; if (hm >= MT_N) hm -= MT_N;
    uint32_t y = (s.x[h] & UPPER_MASK) | (s.x[h1] & LOWER_MASK);
    uint32_t v = s.x[hm] ^ (y >> 1) ^ ((y & 1u) ? MATRIX_A : 0u);
    s.x[h] = v;
    s.h = h1;
    return v;
}

inline bool getBit(const Poly& p, size_t i) { return (p[i >> 6] >> (i & 63)) & 1u; }
inline void setBit(Poly& p, size_t i) { p[i >> 6] |= uint64_t(1) << (i & 63); }

// Characteristic polynomial of MT19937 via Berlekamp-Massey on the LSB of the raw sequence
inline Poly characteristicPolynomial() {
    const int NB = 2 * DEG;
    State s;
    seedState(s, 5489u);
    // R holds the bit sequence reversed: bit (NB-1-n) of R = b_n
    std::vector<uint64_t> R(NB / 64 + 2, 0);
    for (int n = 0; n < NB; ++n) {
        if (step(s) & 1u) R[(NB - 1 - n) >> 6] |= uint64_t(1) << ((NB - 1 - n) & 63);
    }
    const uint64_t* Rp = R.data();
    const size_t RS = R.size();
    const int CW = NB / 64 + 2;
    Poly C(CW, 0), B(CW, 0), T(CW, 0);
    C[0] = 1; B[0] = 1;
    int L = 0, m = 1;
    int degB = 0;                 // upper bound on degree of B
    for (int n = 0; n < NB; ++n) {
        // discrepancy d = sum_{i=0..L} C_i b_{n-i} = parity(<C, R >> (NB-1-n)>)
        const size_t off = static_cast<size_t>(NB - 1 - n);
        const size_t q0 = off >> 6, r = off & 63;
        const int words = L / 64 + 1;
        uint64_t acc = 0;
        for (int w = 0; w < words; ++w) {
            size_t q = q0 + static_cast<size_t>(w);
            uint64_t lo = q < RS ? Rp[q] : 0, hi = (q + 1) < RS ? Rp[q + 1] : 0;
            uint64_t rw = r ? (lo >> r) | (hi << (64 - r)) : lo;
            acc ^= C[w] & rw;
        }
        if (!(__builtin_popcountll(acc) & 1)) { ++m; continue; }
        // C ^= B << m   (B has degree <= degB)
        auto xorShift = [&](Poly& dst, const Poly& src, int sh, int srcDeg) {
            int wq = sh >> 6, rr = sh & 63;
            int top = std::min(static_cast<int>(dst.size()) - 1, (srcDeg + sh) / 64 + 1);
            for (int w = top; w >= wq; --w) {
                uint64_t v = src[w - wq] << rr;
                if (rr && w - wq - 1 >= 0) v |= src[w - wq - 1] >> (64 - rr);
                dst[w] ^= v;
            }
        };
        if (2 * L <= n) {
            T = C;
            int degC = L;
            xorShift(C, B, m, degB);
            L = n + 1 - L;
            B = T;
            degB = degC;
            m = 1;
        } else {
            xorShift(C, B, m, degB);
            ++m;
        }
    }
    if (L != DEG) {
        fprintf(stderr, "mtjump: unexpected minimal polynomial degree %d\n", L);
        std::exit(2);
    }
    Poly phi(PW + 1, 0);   // phi_i = C_{L-i}
    for (int i = 0; i <= L; ++i) if (getBit(C, static_cast<size_t>(L - i))) setBit(phi, static_cast<size_t>(i));
    return phi;
}

// Arithmetic in GF(2)[t] / (phi)
struct Modulus {
    Poly phi;
    std::vector<uint64_t> shifted;   // 64 x TW words: phi << r

    explicit Modulus(Poly p) : phi(std::move(p)), shifted(64 * TW, 0) {
        for (int r = 0; r < 64; ++r) {
            uint64_t* d = &shifted[static_cast<size_t>(r) * TW];
            for (int w = 0; w < TW; ++w) {
                uint64_t v = 0;
                if (w < static_cast<int>(phi.size())) v |= phi[w] << r;
                if (r && w >= 1 && w - 1 < static_cast<int>(phi.size())) v |= phi[w - 1] >> (64 - r);
                d[w] = v;
            }
        }
    }

    // Reduce p (PRODW words) modulo phi in place; afterwards only the low PW words are meaningful
    void reduce(uint64_t* p) const {
        for (int w = PRODW - 1; w >= DEG / 64; --w) {
            // clear bits of word w from the top down, never touching bits below DEG
            const int lowBit = (w == DEG / 64) ? (DEG & 63) : 0;
            uint64_t word = p[w] & ~((lowBit == 0) ? 0ULL : ((uint64_t(1) << lowBit) - 1));
            while (word) {
                int bit = 63 - __builtin_clzll(word);
                int i = w * 64 + bit;
                int sh = i - DEG, wq = sh >> 6, r = sh & 63;
                const uint64_t* src = &shifted[static_cast<size_t>(r) * TW];
                int lim = std::min(TW, PRODW - wq);
                for (int k = 0; k < lim; ++k) p[wq + k] ^= src[k];
                word = p[w] & ~((lowBit == 0) ? 0ULL : ((uint64_t(1) << lowBit) - 1));
            }
        }
        for (int w = PW; w < PRODW; ++w) p[w] = 0;
    }

    // a <- a^2 mod phi (a has PW words, scratch has PRODW words)
    void squareInPlace(Poly& a, uint64_t* scratch) const {
        for (int w = 0; w < PRODW; ++w) scratch[w] = 0;
        for (int w = 0; w < PW; ++w) {
            uint64_t v = a[w];
            uint64_t lo = 0, hi = 0;
            for (int b = 0; b < 32; ++b) lo |= ((v >> b) & 1u) << (2 * b);
            for (int b = 32; b < 64; ++b) hi |= ((v >> b) & 1u) << (2 * (b - 32));
            scratch[2 * w] = lo;
            scratch[2 * w + 1] = hi;
        }
        reduce(scratch);
        for (int w = 0; w < PW; ++w) a[w] = scratch[w];
    }

    // a <- a * t mod phi
    void mulTInPlace(Poly& a, uint64_t* scratch) const {
        for (int w = 0; w < PRODW; ++w) scratch[w] = 0;
        for (int w = PW; w > 0; --w) scratch[w] = (a[w] << 1) | (a[w - 1] >> 63);
        scratch[0] = a[0] << 1;
        reduce(scratch);
        for (int w = 0; w < PW; ++w) a[w] = scratch[w];
    }

    // t^v mod phi (left-to-right binary exponentiation: squarings and shifts only)
    Poly powT(uint64_t v) const {
        Poly r(PW + 1, 0);   // one spare word so mulTInPlace can shift freely
        r[0] = 1;
        if (v == 0) { r.resize(PW); return r; }
        std::vector<uint64_t> scratch(PRODW);
        int top = 63 - __builtin_clzll(v);
        for (int b = top; b >= 0; --b) {
            squareInPlace(r, scratch.data());
            if ((v >> b) & 1u) mulTInPlace(r, scratch.data());
        }
        r.resize(PW);
        return r;
    }

    Poly square(const Poly& a) const {
        Poly r(a);
        r.resize(PW + 1, 0);
        std::vector<uint64_t> scratch(PRODW);
        squareInPlace(r, scratch.data());
        r.resize(PW);
        return r;
    }
};

// s <- g(A) s by Horner's rule; g has degree < DEG
inline void applyPoly(State& s, const Poly& g) {
    // Linearise s so that logical word k is lin[k]
    uint32_t lin[MT_N];
    for (int k = 0; k < MT_N; ++k) {
        int idx = s.h + k; if (idx >= MT_N) idx -= MT_N;
        lin[k] = s.x[idx];
    }
    State r;
    for (int k = 0; k < MT_N; ++k) r.x[k] = 0;
    r.h = 0;
    for (int i = DEG - 1; i >= 0; --i) {
        step(r);
        if (getBit(g, static_cast<size_t>(i))) {
            // r ^= s, respecting r's rotation: r.x[(h+k)%N] ^= lin[k]
            const int h = r.h;
            const int first = MT_N - h;
            uint32_t* __restrict rx = r.x;
            for (int k = 0; k < first; ++k) rx[h + k] ^= lin[k];
            for (int k = first; k < MT_N; ++k) rx[k - first] ^= lin[k];
        }
    }
    s = r;
}

// std::mt19937 positioned at the given state (next output = temper(x_{n+624}))
inline std::mt19937 toEngine(const State& s) {
    std::stringstream ss;
    for (int k = 0; k < MT_N; ++k) {
        int idx = s.h + k; if (idx >= MT_N) idx -= MT_N;
        ss << s.x[idx] << ' ';
    }
    ss << MT_N;
    std::mt19937 e;
    ss >> e;
    return e;
}

// Jump table for offsets that are multiples of a fixed stride:
// powers[k] = t^(stride * 2^k) mod phi, so that a jump by c * stride is the
// composition of the Horner evaluations for the set bits of c.
struct StrideJumper {
    std::vector<Poly> powers;

    StrideJumper(const Modulus& mod, uint64_t stride, uint64_t maxMultiple) {
        Poly p = mod.powT(stride);
        powers.push_back(p);
        for (uint64_t m = maxMultiple >> 1; m > 0; m >>= 1) {
            p = mod.square(p);
            powers.push_back(p);
        }
    }

    // Engine equivalent to std::mt19937(seed) followed by discard(multiple * stride)
    std::mt19937 engineAt(uint32_t seed, uint64_t multiple) const {
        State s;
        seedState(s, seed);
        for (size_t k = 0; k < powers.size() && (multiple >> k) != 0; ++k) {
            if ((multiple >> k) & 1u) applyPoly(s, powers[k]);
        }
        return toEngine(s);
    }
};

} // namespace mtjump

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

    return octree.applyToTris(from, to, [&](size_t idx, const Triangle& tri) {
        if (idx == srcTriIdx || idx == dstTriIdx) return false;

        val_t dist = rayTriangleIntersect(from, dirNorm, tri.a, tri.b, tri.c);
        if (dist > EPSILON && dist < rayLen - EPSILON) {
            return true;  // Ray is blocked
        }
        return false;
    });
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
    FlatOctree flatOctree;          // Contiguous copy of the octree used for ray queries

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

    // Initialize areas (also spins up the OpenMP thread pool)
    state.areas.resize(state.numTriangles);
    #pragma omp parallel for schedule(static)
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

// Number of raw mt19937 draws consumed by one RandomGenerator::rand() call,
// determined against the standard library rather than assumed.
static uint64_t drawsPerRandCall() {
    std::mt19937 a(42);
    std::uniform_real_distribution<val_t> dist(0.0f, 1.0f);
    dist(a);
    for (uint64_t k = 1; k <= 8; ++k) {
        std::mt19937 b(42);
        b.discard(k);
        if (b == a) return k;
    }
    fprintf(stderr, "Unable to determine RNG draws per call\n");
    std::exit(2);
}

// Range of chunks [next, end) owned by one worker, packed into one atomic so
// that the owner (taking from the front) and thieves (taking from the back)
// can claim chunks with a single compare-and-swap.
struct alignas(64) ChunkRange {
    std::atomic<uint64_t> packed{0};
    static uint64_t pack(uint32_t next, uint32_t end) { return (static_cast<uint64_t>(next) << 32) | end; }
    static uint32_t next(uint64_t v) { return static_cast<uint32_t>(v >> 32); }
    static uint32_t end(uint64_t v) { return static_cast<uint32_t>(v & 0xffffffffu); }

    // Owner: claim the next chunk, or return false if the range is exhausted
    bool claimFront(uint32_t& chunk) {
        uint64_t v = packed.load(std::memory_order_relaxed);
        while (next(v) < end(v)) {
            if (packed.compare_exchange_weak(v, pack(next(v) + 1, end(v)),
                                             std::memory_order_acq_rel, std::memory_order_relaxed)) {
                chunk = next(v);
                return true;
            }
        }
        return false;
    }
    uint32_t remaining() const {
        uint64_t v = packed.load(std::memory_order_relaxed);
        return next(v) < end(v) ? end(v) - next(v) : 0;
    }
    // Thief: take the back half of the remaining range; returns false if nothing was taken
    bool stealBack(uint32_t& from, uint32_t& to) {
        uint64_t v = packed.load(std::memory_order_relaxed);
        while (next(v) < end(v)) {
            uint32_t rem = end(v) - next(v);
            uint32_t mid = end(v) - (rem + 1) / 2;
            if (packed.compare_exchange_weak(v, pack(next(v), mid),
                                             std::memory_order_acq_rel, std::memory_order_relaxed)) {
                from = mid;
                to = end(v);
                return true;
            }
        }
        return false;
    }
};

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const size_t n = state.numTriangles;
    const std::vector<Triangle>& tris = state.triangles;

    // The sequential reference draws all samples from one mt19937(42) stream, in
    // row-major pair order.  Each pair that passes the normal-culling test
    // consumes exactly NUM_RAYS * 4 rand() calls, culled pairs consume none.
    // We therefore count the sampled pairs per row, split the pair sequence into
    // chunks of equal sample count and position an independent generator at the
    // exact stream offset of every chunk, reproducing the reference samples bit
    // for bit.  Each worker owns a contiguous range of chunks (its generator
    // simply flows from one chunk into the next); idle workers steal the back
    // half of the largest remaining range and jump their generator there.
    const uint64_t drawsPerPair = static_cast<uint64_t>(NUM_RAYS) * 4 * drawsPerRandCall();

    std::vector<uint64_t> rowPrefix(n + 1, 0);   // sampled pairs in rows [0, i)
    uint64_t totalSampled = 0;
    uint64_t numChunks = 0, pairsPerChunk = 0;
    bool useJump = false;
    std::vector<size_t> chunkStart;
    std::unique_ptr<mtjump::StrideJumper> jumper;
    std::vector<ChunkRange> ranges;
    std::atomic<size_t> rowsDone{0};

    auto isSampled = [&](size_t i, const Vec3& ni, size_t j) {
        return i != j && !(ni.dot(tris[j].normal()) > 0.99f);   // same test as computeKij
    };

    // Cell index (row-major, including the diagonal) of the sampled pair with the given rank
    auto cellOfRank = [&](uint64_t rank) -> size_t {
        if (rank >= totalSampled) return n * n;
        size_t i = static_cast<size_t>(std::upper_bound(rowPrefix.begin(), rowPrefix.end(), rank) - rowPrefix.begin()) - 1;
        uint64_t seen = rowPrefix[i];
        const Vec3 ni = tris[i].normal();
        for (size_t j = 0; j < n; ++j) {
            if (!isSampled(i, ni, j)) continue;
            if (seen == rank) return i * n + j;
            ++seen;
        }
        return (i + 1) * n;   // unreachable
    };

    #pragma omp parallel
    {
        const int numThreads = omp_get_num_threads();
        const int tid = omp_get_thread_num();

        // --- 1. count sampled pairs per row -----------------------------------
        #pragma omp for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const Vec3 ni = tris[i].normal();
            uint64_t cnt = 0;
            for (size_t j = 0; j < n; ++j) cnt += isSampled(i, ni, j) ? 1 : 0;
            rowPrefix[i + 1] = cnt;
        }

        #pragma omp single
        {
            for (size_t i = 0; i < n; ++i) rowPrefix[i + 1] += rowPrefix[i];
            totalSampled = rowPrefix[n];

            // Fine-grained chunks (about 1000 sampled pairs each, within limits) for
            // good load balancing; the generator only has to jump after a steal.
            const uint64_t P = static_cast<uint64_t>(numThreads);
            numChunks = std::clamp<uint64_t>(totalSampled / 1000ULL, P, 64 * P);
            if (numChunks > totalSampled) numChunks = std::max<uint64_t>(1, totalSampled);
            pairsPerChunk = (totalSampled + numChunks - 1) / numChunks;
            chunkStart.assign(numChunks + 1, n * n);

            // For small streams a plain discard() is cheaper than the polynomial setup.
            useJump = totalSampled * drawsPerPair > 30000000ULL;

            ranges = std::vector<ChunkRange>(static_cast<size_t>(numThreads));
            for (uint64_t t = 0; t < P; ++t) {
                uint32_t from = static_cast<uint32_t>(t * numChunks / P);
                uint32_t to = static_cast<uint32_t>((t + 1) * numChunks / P);
                ranges[t].packed.store(ChunkRange::pack(from, to), std::memory_order_relaxed);
            }
        }

        // --- 2. chunk boundaries, and jump tables on one thread -----------------
        #pragma omp single nowait
        {
            if (useJump) {
                mtjump::Modulus modulus(mtjump::characteristicPolynomial());
                jumper = std::make_unique<mtjump::StrideJumper>(modulus, pairsPerChunk * drawsPerPair, numChunks);
            }
        }
        #pragma omp for schedule(dynamic, 16)
        for (uint64_t c = 0; c < numChunks; ++c) chunkStart[c] = cellOfRank(c * pairsPerChunk);
        // (implicit barrier: both the boundaries and the jump tables are complete here)

        // --- 3. form factors with work stealing ---------------------------------
        RandomGenerator rng(std::mt19937(42u));
        uint64_t rngChunk = 0;          // chunk at whose start rng is positioned

        auto processChunk = [&](uint32_t c) {
            const size_t qa = chunkStart[c];
            const size_t qb = chunkStart[c + 1];
            if (c != rngChunk) {
                std::mt19937 engine;
                if (useJump) {
                    engine = jumper->engineAt(42u, c);
                } else {
                    engine.seed(42u);
                    engine.discard(c * pairsPerChunk * drawsPerPair);
                }
                rng = RandomGenerator(engine);
            }
            size_t i = qa / n;
            size_t j = qa % n;
            for (size_t q = qa; q < qb; ++q) {
                if (i != j) {
                    state.kij[q] = computeKij(i, j, tris, state.flatOctree, rng);
                }
                if (++j == n) {
                    j = 0;
                    ++i;
                    size_t done = rowsDone.fetch_add(1, std::memory_order_relaxed) + 1;
                    if (done % 100 == 0 || done == n) {
                        printf("  Progress: %zu/%zu triangles\n", done, n);
                    }
                }
            }
            rngChunk = c + 1;
        };

        ChunkRange& mine = ranges[static_cast<size_t>(tid)];
        for (;;) {
            uint32_t c;
            if (mine.claimFront(c)) {
                processChunk(c);
                continue;
            }
            // Own range exhausted: steal the back half of the largest remaining range
            int victim = -1;
            uint32_t best = 0;
            for (int t = 0; t < numThreads; ++t) {
                uint32_t rem = ranges[static_cast<size_t>(t)].remaining();
                if (rem > best) { best = rem; victim = t; }
            }
            if (victim < 0) break;
            uint32_t from, to;
            if (ranges[static_cast<size_t>(victim)].stealBack(from, to)) {
                mine.packed.store(ChunkRange::pack(from, to), std::memory_order_release);
            }
        }
    }
}

// Threads to use for a phase with the given number of elementary operations,
// so that tiny problems are not dominated by fork/join and barrier overhead.
static int threadsForWork(double ops, double opsPerThread) {
    int maxT = omp_get_max_threads();
    double want = ops / opsPerThread;
    if (want < 1.0) return 1;
    if (want >= static_cast<double>(maxT)) return maxT;
    return static_cast<int>(want);
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

    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;
    const int* __restrict tau = state.tau.data();
    const val_t* __restrict kijM = state.kij.data();
    const val_t* __restrict areas = state.areas.data();
    const val_t* __restrict rho = state.rho.data();
    const val_t* __restrict radE = state.radE.data();
    val_t* __restrict radB = state.radB.data();

    // Within a timestep all receivers are independent: radB[t, i] only reads
    // radB[t - tau, j] with tau >= 1 (distinct triangle centres), i.e. rows that
    // were completed in earlier timesteps.  The j-summation order per receiver is
    // unchanged, so results are bit-identical to the sequential version.
    const int nThreads = threadsForWork(static_cast<double>(n) * static_cast<double>(n), 16384.0);
    #pragma omp parallel num_threads(nThreads)
    {
        for (size_t t = 0; t < T; ++t) {
            #pragma omp for schedule(static)
            for (size_t i = 0; i < n; ++i) {
                val_t sumB = ZERO;
                const int* tauRow = tau + i * n;
                const val_t* kijRow = kijM + i * n;

                for (size_t j = 0; j < n; ++j) {
                    if (i == j) continue;

                    int tauij = tauRow[j];

                    // Skip if wave hasn't yet propagated from j to i
                    if (static_cast<int>(t) < tauij) continue;

                    val_t kij = kijRow[j];
                    if (kij <= ZERO) continue;

                    // Get radiosity from source triangle at time when emission occurred
                    size_t srcTime = t - static_cast<size_t>(tauij);
                    val_t radJ = radB[srcTime * n + j];
                    if (radJ <= ZERO) continue;

                    // Accumulate contribution: form factor * area * source radiosity
                    sumB += std::min(kij * areas[j], ONE) * radJ;
                }

                // Update radiosity: reflection + emission
                radB[t * n + i] = rho[i] * sumB + radE[t * n + i];
            }
            // (implicit barrier of the worksharing loop orders the timesteps)

            #pragma omp master
            if ((t + 1) % 10 == 0 || t + 1 == T) {
                printf("  Timestep %zu/%zu\n", t + 1, T);
            }
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const size_t n = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Contiguous copy of the source triangle's radiosity time series
    std::vector<val_t> srcSeries(T);
    for (size_t tt = 0; tt < T; ++tt) srcSeries[tt] = state.radB[state.idxTN(tt, state.sourceIndex)];

    // Same team size as the simulation phase (avoids re-sizing the thread team)
    const int nThreads = threadsForWork(static_cast<double>(n) * static_cast<double>(n), 16384.0);
    #pragma omp parallel for schedule(dynamic, 16) num_threads(nThreads)
    for (size_t i = 0; i < n; ++i) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = state.radB[tt * n + i];
                val_t pS = srcSeries[tt - t];
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
    #pragma omp parallel for schedule(static) reduction(+:receivedEnergy)
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
    #pragma omp parallel for schedule(static) reduction(+:nonZeroKij)
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
            return 1;
        }
    }

    return 0;
}
