/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed implementation of room impulse response
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
#include <bit>
#include <bitset>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <map>
#include <memory>
#include <random>
#include <vector>

#include <mpi.h>

#include "../common/results_output.hpp"

// ============================================================================
// Types and Constants
// ============================================================================

using idx_t = uint32_t;
using val_t = float;

constexpr val_t PI = 3.14159265358979323846f;
constexpr val_t ZERO = 0.0f;
constexpr val_t ONE = 1.0f;
constexpr val_t ROOM_RADIUS = 10.0f;
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

// MT19937 with access to its linear state transition for jumping to the
// beginning of a rank's part of the original stream.
class JumpableMT19937 {
    static constexpr size_t Words = 624;
    static constexpr size_t Shift = 397;
    static constexpr size_t Degree = 19937;
    std::array<uint32_t, Words> state;
    size_t next = 0;

    static uint32_t twist(uint32_t first, uint32_t second, uint32_t shifted) {
        const uint32_t bits = (first & 0x80000000u) | (second & 0x7fffffffu);
        return shifted ^ (bits >> 1) ^ ((bits & 1u) * 0x9908b0dfu);
    }

    void refill() {
        for (size_t i = 0; i < Words - Shift; ++i) {
            state[i] = twist(state[i], state[i + 1], state[i + Shift]);
        }
        for (size_t i = Words - Shift; i < Words - 1; ++i) {
            state[i] = twist(state[i], state[i + 1], state[i + Shift - Words]);
        }
        state.back() = twist(state.back(), state.front(), state[Shift - 1]);
        next = 0;
    }

    static std::bitset<2 * Degree> jumpPolynomial(uint64_t offset) {
        // Exponents of the non-leading terms of the MT19937 characteristic
        // polynomial over GF(2), from Berlekamp–Massey on its output bits.
        constexpr int exponents[] = {
            0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681,
            3908, 4135, 4362, 4753, 5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752,
            7979, 8206, 9505, 9901, 9969, 10128, 10693, 10761, 10920, 11089, 11147, 11157,
            11215, 11321, 11374, 11384, 11485, 11611, 11712, 11717, 11838, 11881, 11944, 11997,
            12277, 12335, 12393, 12504, 12509, 12620, 12673, 12731, 12736, 12789, 12905, 12958,
            12963, 13137, 13185, 13190, 13243, 13301, 13412, 13528, 13533, 13639, 13697, 13760,
            13813, 13866, 14093, 14151, 14209, 14320, 14325, 14436, 14547, 14552, 14605, 14721,
            14774, 14779, 14953, 15001, 15006, 15059, 15117, 15228, 15344, 15349, 15455, 15513,
            15576, 15629, 15682, 15909, 15967, 16025, 16136, 16141, 16252, 16363, 16368, 16421,
            16537, 16590, 16595, 16817, 16822, 16875, 16933, 17044, 17160, 17271, 17329, 17445,
            17498, 17725, 17783, 17841, 17952, 18068, 18179, 18237, 18406, 18633, 18691, 18860,
            19087, 19314,
        };

        // Compute x^offset modulo the characteristic polynomial. Squaring in
        // GF(2) doubles each exponent and introduces no cross terms.
        std::bitset<2 * Degree> polynomial{1};
        for (uint64_t bit = std::bit_floor(offset); bit != 0; bit >>= 1) {
            std::bitset<2 * Degree> power;
            const size_t shift = (offset & bit) != 0;
            for (size_t i = 0; i < Degree; ++i) {
                power[2 * i + shift] = polynomial[i];
            }
            for (size_t i = 2 * Degree; i-- > Degree;) {
                if (!power[i]) continue;
                power.reset(i);
                for (int exponent : exponents) {
                    power.flip(i - Degree + exponent);
                }
            }
            polynomial = power;
        }
        return polynomial;
    }

public:
    using result_type = uint32_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT32_MAX; }

    explicit JumpableMT19937(uint32_t seed) {
        state[0] = seed;
        for (uint32_t i = 1; i < Words; ++i) {
            state[i] = 1812433253u * (state[i - 1] ^ (state[i - 1] >> 30)) + i;
        }
        refill();
    }

    // Called on the initial state, before any draws.
    void skip(uint64_t offset) {
        if (offset < 1'000'000) {
            while (offset >= Words) {
                refill();
                offset -= Words;
            }
            next = offset;
            return;
        }

        const auto polynomial = jumpPolynomial(offset);
        std::array<uint32_t, Degree + Words - 1> sequence;
        std::copy(state.begin(), state.end(), sequence.begin());
        for (size_t i = Words; i < sequence.size(); ++i) {
            sequence[i] = twist(sequence[i - Words], sequence[i - Words + 1],
                                sequence[i - Words + Shift]);
        }
        state.fill(0);
        for (size_t i = 0; i < Degree; ++i) {
            if (!polynomial[i]) continue;
            for (size_t j = 0; j < Words; ++j) {
                state[j] ^= sequence[i + j];
            }
        }
    }

    result_type operator()() {
        if (next == Words) refill();
        uint32_t value = state[next++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680u;
        value ^= (value << 15) & 0xefc60000u;
        return value ^ (value >> 18);
    }
};

class RandomGenerator {
    JumpableMT19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint64_t start = 0) : rng(42), dist(0.0f, 1.0f) {
        // Each uniform float uses one MT19937 word.
        rng.skip(start);
    }
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
                  const Octree& octree,
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
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

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

int computeTau(const Vec3& centerI, const Vec3& centerJ) {
    val_t dist = (centerI - centerJ).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// Each rank owns a contiguous block of receiver triangles in serial order.
struct TriangleBlock {
    MPI_Comm comm;
    int rank, ranks;
    std::vector<int> counts, displacements;
    int begin, size;

    TriangleBlock(MPI_Comm comm, int triangles) : comm(comm) {
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        counts.resize(ranks);
        displacements.resize(ranks);
        int offset = 0;
        for (int peer = 0; peer < ranks; ++peer) {
            counts[peer] = triangles / ranks + (peer < triangles % ranks);
            displacements[peer] = offset;
            offset += counts[peer];
        }
        begin = displacements[rank];
        size = counts[rank];
    }

    bool root() const { return rank == 0; }
};

struct Interaction {
    int historyOffset;
    val_t weight;
};

struct SimulationState {
    int numTriangles;
    int numTimesteps;
    int sourceIndex;
    val_t reflectivity;
    std::vector<std::vector<Interaction>> interactions;
    std::vector<val_t> localRadiosity;   // Triangle-major, for cross-correlation
    std::vector<val_t> sourceRadiosity;
    std::vector<val_t> distances;
    uint64_t nonZeroKij = 0;
    int minDelay = std::numeric_limits<int>::max();
    int maxDelay = 0;
};

bool samplesPair(const Vec3& normalI, const Vec3& normalJ, int i, int j) {
    // The serial generator advances only for pairs that survive this cull.
    return i != j && normalI.dot(normalJ) <= 0.99f;
}

void precompute(SimulationState& state, const TriangleBlock& block,
                const std::vector<Triangle>& triangles, const Octree& octree) {
    if (block.root()) printf("Computing time delays (Tau) and form factors (Kij)...\n");

    const int n = state.numTriangles;
    std::vector<Vec3> normals(n), centers(n);
    std::vector<val_t> areas(n);
    for (int i = 0; i < n; ++i) {
        normals[i] = triangles[i].normal();
        centers[i] = triangles[i].center();
        areas[i] = triangles[i].area();
    }

    // Save the exact cull decisions used to calculate the stream offsets.
    // A row is encoded as a bit set so the second pass keeps the same decisions
    // even when the compiler vectorizes the normal-dot-product loop.
    const size_t wordsPerRow = (n + 63) / 64;
    std::vector<uint64_t> sampled(static_cast<size_t>(block.size) * wordsPerRow);
    uint64_t pairs = 0;
    for (int local = 0; local < block.size; ++local) {
        const int i = block.begin + local;
        uint64_t* row = sampled.data() + local * wordsPerRow;
        for (int j = 0; j < n; ++j) {
            if (samplesPair(normals[i], normals[j], i, j)) {
                row[j / 64] |= uint64_t{1} << (j % 64);
                ++pairs;
            }
        }
    }

    uint64_t precedingPairs = 0;
    MPI_Exscan(&pairs, &precedingPairs, 1, MPI_UINT64_T, MPI_SUM, block.comm);
    if (block.root()) precedingPairs = 0;
    RandomGenerator rng(precedingPairs * (4 * NUM_RAYS));

    state.interactions.resize(block.size);
    for (int local = 0; local < block.size; ++local) {
        const int i = block.begin + local;
        auto& links = state.interactions[local];
        if (state.numTimesteps > 2 * ROOM_RADIUS * INV_WAVE_SPEED) links.reserve(n);
        const uint64_t* row = sampled.data() + local * wordsPerRow;
        for (size_t word = 0; word < wordsPerRow; ++word) {
            uint64_t bits = row[word];
            while (bits != 0) {
                const int j = static_cast<int>(word * 64 + std::countr_zero(bits));
                bits &= bits - 1;
                const val_t kij = computeKij(i, j, triangles, octree, rng);
                state.nonZeroKij += kij > EPSILON;
                if (kij <= ZERO) continue;
                const int delay = computeTau(centers[i], centers[j]);
                if (delay >= state.numTimesteps) continue;

                // Distinct icosphere triangles have distinct centers, so
                // every interaction reads from an earlier timestep.
                state.minDelay = std::min(state.minDelay, delay);
                state.maxDelay = std::max(state.maxDelay, delay);
                links.push_back({j - delay * n, std::min(kij * areas[j], ONE)});
            }
        }
    }

    // A batch can advance as many steps as the shortest interaction delay.
    int delays[] = {-state.minDelay, state.maxDelay};
    MPI_Allreduce(MPI_IN_PLACE, delays, 2, MPI_INT, MPI_MAX, block.comm);
    state.minDelay = -delays[0];
    state.maxDelay = delays[1];
}

void runSimulation(SimulationState& state, const TriangleBlock& block) {
    if (block.root()) printf("Running wave propagation simulation...\n");
    const int n = state.numTriangles;
    const int steps = state.numTimesteps;
    const int historyRows = state.maxDelay + 1;
    const int batchSize = std::min(state.minDelay, historyRows);

    state.localRadiosity.resize(static_cast<size_t>(steps) * block.size);
    state.sourceRadiosity.resize(steps);
    if (steps == 0) return;

    // Mirror a rolling window so each interaction can address its past value
    // with a precomputed offset, including when the window wraps.
    const size_t historySpan = static_cast<size_t>(n) * historyRows;
    std::vector<val_t> history(2 * historySpan);
    const int bufferedSteps = batchSize > 1 ? batchSize : 0;
    std::vector<val_t> outgoing(static_cast<size_t>(bufferedSteps) * block.size);
    std::vector<val_t> incoming(static_cast<size_t>(bufferedSteps) * n);
    std::vector<int> counts(block.ranks), displacements(block.ranks);
    const bool evenBlocks = n % block.ranks == 0;
    const int localSource = state.sourceIndex - block.begin;

    for (int first = 0; first < steps;) {
        const int width = std::min(batchSize, steps - first);
        // A single timestep can be exchanged in place in the history row.
        const bool inPlace = width == 1;
        val_t* receive = inPlace ? history.data() + (first % historyRows) * n : incoming.data();
        for (int k = 0; k < width; ++k) {
            const int t = first + k;
            const val_t* past = history.data() + historySpan + (t % historyRows) * n;
            val_t* send = inPlace ? receive + block.begin : outgoing.data() + k * block.size;
            for (int local = 0; local < block.size; ++local) {
                val_t sumB = ZERO;
                for (const auto& link : state.interactions[local]) {
                    const val_t radJ = past[link.historyOffset];
                    if (radJ <= ZERO) continue;
                    sumB += link.weight * radJ;
                }

                const val_t emission = local == localSource && t < steps / 2 ? ONE : ZERO;
                const val_t radB = state.reflectivity * sumB + emission;
                state.localRadiosity[static_cast<size_t>(local) * steps + t] = radB;
                send[local] = radB;
            }
        }

        // Each rank sends its own contiguous block for every timestep in the
        // batch. Reassemble the full timestep rows in the rolling window.
        const int sendCount = width * block.size;
        const void* send = inPlace ? MPI_IN_PLACE : outgoing.data();
        if (evenBlocks) {
            MPI_Allgather(send, sendCount, MPI_FLOAT, receive, sendCount, MPI_FLOAT, block.comm);
        } else {
            for (int peer = 0; peer < block.ranks; ++peer) {
                counts[peer] = width * block.counts[peer];
                displacements[peer] = width * block.displacements[peer];
            }
            MPI_Allgatherv(send, sendCount, MPI_FLOAT, receive,
                           counts.data(), displacements.data(), MPI_FLOAT, block.comm);
        }
        for (int k = 0; k < width; ++k) {
            const int t = first + k;
            val_t* row = history.data() + (t % historyRows) * n;
            if (!inPlace) {
                for (int peer = 0; peer < block.ranks; ++peer) {
                    const int count = block.counts[peer];
                    const int begin = block.displacements[peer];
                    std::copy_n(receive + width * begin + k * count, count, row + begin);
                }
            }
            std::copy_n(row, n, row + historySpan);
            state.sourceRadiosity[t] = row[state.sourceIndex];
            if (block.root() && ((t + 1) % 10 == 0 || t + 1 == steps)) {
                printf("  Timestep %d/%d\n", t + 1, steps);
            }
        }
        first += width;
    }
}

void computeDistances(SimulationState& state, const TriangleBlock& block) {
    if (block.root()) printf("Computing distances via cross-correlation...\n");
    const int steps = state.numTimesteps;
    state.distances.resize(block.size, ZERO);
    if (steps == 0) return;

    const val_t* source = state.sourceRadiosity.data();
    for (int i = 0; i < block.size; ++i) {
        const val_t* trace = state.localRadiosity.data() + static_cast<size_t>(i) * steps;
        val_t maxCorr = ZERO;
        int bestT = 0;

        for (int t = 0; t < steps; ++t) {
            val_t sum = ZERO;
            for (int tt = 0; tt < steps - t; ++tt) {
                sum += source[tt] * trace[t + tt];
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = t;
            }
        }

        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}

uint64_t computeHash(const SimulationState& state, const TriangleBlock& block) {
    uint64_t hash = 0;
    for (int i = 0; i < block.size; ++i) {
        const uint64_t bits = std::bit_cast<uint32_t>(state.distances[i]);
        hash ^= (bits + block.begin + i) * 0x9e3779b97f4a7c15ULL;
    }
    return hash;
}

uint64_t countReceivingEnergy(const SimulationState& state, const TriangleBlock& block) {
    uint64_t received = 0;
    auto first = state.localRadiosity.begin();
    for (int i = 0; i < block.size; ++i) {
        const auto end = first + state.numTimesteps;
        received += std::any_of(first, end, [](val_t value) { return value > EPSILON; });
        first = end;
    }
    return received;
}

bool validateResults(const SimulationState& state, const std::vector<val_t>& distances,
                     uint64_t receivedEnergy, uint64_t nonZeroKij) {
    printf("\nValidation:\n");

    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < distances.size(); ++i) {
        const val_t d = distances[i];
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
        nonZeroCount += d > EPSILON;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%d\n", nonZeroCount, state.numTriangles);

    const val_t srcDist = distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    printf("  Triangles receiving energy: %" PRIu64 "/%d\n", receivedEnergy, state.numTriangles);
    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    const uint64_t pairs = static_cast<uint64_t>(state.numTriangles) * state.numTriangles;
    printf("  Non-zero form factors: %" PRIu64 "/%" PRIu64 " (%.2f%%)\n",
           nonZeroKij, pairs, 100.0f * nonZeroKij / static_cast<val_t>(pairs));
    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) return false;

    printf("  Validation: PASSED\n");
    return true;
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
    printf("               20, 80, 320, 1280, 5120, 20480, 81920\n");
    printf("  -t <num>     Number of timesteps (default: 50)\n");
    printf("  -s <num>     Source triangle index (default: 0)\n");
    printf("  -r <val>     Reflectivity 0.0-1.0 (default: 0.8)\n");
    printf("  -v           Enable validation\n");
    printf("  -o           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int runBenchmark(int argc, char** argv) {
    int worldRank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
            if (worldRank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (worldRank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (timesteps < 0) {
        if (worldRank == 0) fprintf(stderr, "Timesteps must be non-negative.\n");
        return 1;
    }

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    const int n = 20 << (2 * subdivisions);
    MPI_Comm comm = MPI_COMM_WORLD;
    if (worldSize > n) {
        // Exclude ranks that have no triangle rows to compute.
        MPI_Comm_split(MPI_COMM_WORLD, worldRank < n ? 0 : MPI_UNDEFINED, worldRank, &comm);
    }

    int status = 0;
    if (comm != MPI_COMM_NULL) {
        const TriangleBlock block(comm, n);
        if (block.root()) {
            printf("Room Response Simulation Benchmark\n");
            printf("===================================\n");
            printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
            printf("Timesteps: %d\n", timesteps);
            printf("Source triangle: %d\n", sourceIdx);
            printf("Reflectivity: %.2f\n", reflectivity);
            printf("Validation: %s\n", validate ? "enabled" : "disabled");
            printf("\n");
        }

        SimulationState state;
        state.numTriangles = n;
        state.numTimesteps = timesteps;
        state.sourceIndex = static_cast<int>(static_cast<size_t>(sourceIdx) % n);
        state.reflectivity = reflectivity;

        double durations[3];
        {
            const IcosphereMesh mesh(subdivisions, ROOM_RADIUS);
            Octree octree;
            if (block.root()) {
                printf("Generated icosphere mesh with %zu triangles\n", mesh.triangles.size());
                printf("Building octree...\n\n");
            }
            octree.build(mesh.triangles);

            MPI_Barrier(comm);
            const double start = MPI_Wtime();
            precompute(state, block, mesh.triangles, octree);
            durations[0] = MPI_Wtime() - start;
        }

        const double startSim = MPI_Wtime();
        runSimulation(state, block);
        durations[1] = MPI_Wtime() - startSim;

        const double startDist = MPI_Wtime();
        computeDistances(state, block);
        durations[2] = MPI_Wtime() - startDist;

        double phaseTimes[3];
        MPI_Reduce(durations, phaseTimes, 3, MPI_DOUBLE, MPI_MAX, 0, comm);

        const uint64_t localHash = computeHash(state, block);
        uint64_t hash = 0;
        MPI_Reduce(&localHash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

        // Report aggregate storage for the numerical arrays, including the
        // rolling history and the buffers used for each radiosity exchange.
        uint64_t localMemory = 0;
        for (const auto& links : state.interactions) {
            localMemory += links.capacity() * sizeof(Interaction);
        }
        const uint64_t historyRows = state.maxDelay + 1;
        const uint64_t batchSize = std::min(state.minDelay, state.maxDelay + 1);
        if (timesteps > 0) localMemory += 2 * n * historyRows * sizeof(val_t);
        if (timesteps > 0 && batchSize > 1) {
            localMemory += (n + block.size) * batchSize * sizeof(val_t);
        }
        localMemory += (state.localRadiosity.size() + state.sourceRadiosity.size()
                        + state.distances.size()) * sizeof(val_t);

        const uint64_t localStats[] = {
            localMemory, state.nonZeroKij, validate ? countReceivingEnergy(state, block) : 0
        };
        uint64_t stats[3];
        MPI_Reduce(localStats, stats, 3, MPI_UINT64_T, MPI_SUM, 0, comm);

        std::vector<val_t> distances;
        if (printResults || validate) {
            if (block.root()) distances.resize(n);
            MPI_Gatherv(state.distances.data(), block.size, MPI_FLOAT, distances.data(),
                        block.counts.data(), block.displacements.data(), MPI_FLOAT, 0, comm);
        }

        if (block.root()) {
            const long preTime = static_cast<long>(1000 * phaseTimes[0]);
            const long simTime = static_cast<long>(1000 * phaseTimes[1]);
            const long distTime = static_cast<long>(1000 * phaseTimes[2]);
            const long totalTime = preTime + simTime + distTime;
            printf("Precomputation time: %ld ms\n\n", preTime);
            printf("Simulation time: %ld ms\n\n", simTime);
            printf("Distance computation time: %ld ms\n\n", distTime);
            printf("Total computation time: %ld ms\n", totalTime);

            printf("\nPerformance:\n");
            printf("  Triangles: %d\n", n);
            printf("  Timesteps: %d\n", timesteps);
            printf("  Form factor computations: %.2e\n", static_cast<double>(n) * n);
            printf("  Simulation operations: %.2e\n", static_cast<double>(n) * n * timesteps);
            printf("  Distance computations: %.2e\n", static_cast<double>(n) * timesteps * timesteps);
            printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
            printf("  Memory usage: %.2f MB\n", stats[0] / (1024.0 * 1024.0));
            printf("  Result hash: %016" PRIX64 "\n\n", hash);

            if (printResults) {
                const std::vector<double> distData(distances.begin(), distances.end());
                print_results(distData, "Distances");
            }
            if (validate && !validateResults(state, distances, stats[2], stats[1])) {
                status = 1;
            }
        }
        if (comm != MPI_COMM_WORLD) MPI_Comm_free(&comm);
    }

    if (validate) MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int status = 1;
    try {
        status = runBenchmark(argc, argv);
    } catch (const std::exception& error) {
        int rank;
        MPI_Comm_rank(MPI_COMM_WORLD, &rank);
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return status;
}
