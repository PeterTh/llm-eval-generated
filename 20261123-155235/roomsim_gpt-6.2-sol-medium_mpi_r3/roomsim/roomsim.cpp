/**
 * Room Response Simulation Benchmark
 * 
 * This is a distributed MPI implementation of room impulse response
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
constexpr val_t WAVE_SPEED = 0.5f;        // Wave propagation speed (units per timestep)
constexpr val_t INV_WAVE_SPEED = 1.0f / WAVE_SPEED;
constexpr val_t ROOM_RADIUS = 10.0f;
// The extra step accounts for rounding in mesh generation and the distance.
constexpr int MAX_DELAY = static_cast<int>(2 * ROOM_RADIUS * INV_WAVE_SPEED) + 1;
static_assert(MAX_DELAY <= std::numeric_limits<uint8_t>::max());
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

struct RaySegment {
    Vec3 halfDirection, midpoint, absHalfDirection;

    RaySegment(const Vec3& origin, const Vec3& direction)
        : halfDirection(direction * 0.5f), midpoint(origin + halfDirection),
          absHalfDirection(std::abs(halfDirection.x), std::abs(halfDirection.y),
                           std::abs(halfDirection.z)) {}
};

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
    bool rayIntersectsBox(const RaySegment& ray) const {
        const Vec3& d = ray.halfDirection;
        const Vec3& ad = ray.absHalfDirection;
        Vec3 c = ray.midpoint - center;

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
    bool applyToTris(const RaySegment& ray, Func&& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(ray)) {
                if (children[i]->applyToTris(ray, func)) return true;
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

// MT19937 with a jump to an absolute position in the seed-42 stream.
// Its transition is linear over GF(2), so a large jump can be computed without
// generating or communicating the intervening draws.
class Mt19937 {
    static constexpr int stateSize = 624;
    static constexpr int shiftSize = 397;
    static constexpr int degree = 19937;

    // Nonzero powers in the characteristic polynomial of the one-draw
    // transition. These can be obtained by applying Berlekamp-Massey to
    // 2 * degree successive low bits from the standard MT19937 stream.
    static constexpr int polynomialPowers[] = {
        0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454,
        3681, 3908, 4135, 4362, 4753, 5661, 6337, 6569, 7129, 7477, 7525,
        7583, 7752, 7979, 8206, 9505, 9901, 9969, 10128, 10693, 10761, 10920,
        11089, 11147, 11157, 11215, 11321, 11374, 11384, 11485, 11611, 11712, 11717,
        11838, 11881, 11944, 11997, 12277, 12335, 12393, 12504, 12509, 12620, 12673,
        12731, 12736, 12789, 12905, 12958, 12963, 13137, 13185, 13190, 13243, 13301,
        13412, 13528, 13533, 13639, 13697, 13760, 13813, 13866, 14093, 14151, 14209,
        14320, 14325, 14436, 14547, 14552, 14605, 14721, 14774, 14779, 14953, 15001,
        15006, 15059, 15117, 15228, 15344, 15349, 15455, 15513, 15576, 15629, 15682,
        15909, 15967, 16025, 16136, 16141, 16252, 16363, 16368, 16421, 16537, 16590,
        16595, 16817, 16822, 16875, 16933, 17044, 17160, 17271, 17329, 17445, 17498,
        17725, 17783, 17841, 17952, 18068, 18179, 18237, 18406, 18633, 18691, 18860,
        19087, 19314, degree
    };

    std::array<uint32_t, stateSize> words;
    int next = 0;

    void advance(int i, int following, int shifted) {
        uint32_t joined = (words[i] & 0x80000000u) | (words[following] & 0x7fffffffu);
        words[i] = words[shifted] ^ (joined >> 1) ^ ((joined & 1) ? 0x9908b0dfu : 0);
    }

    void twist() {
        for (int i = 0; i < stateSize - shiftSize; ++i) {
            advance(i, i + 1, i + shiftSize);
        }
        for (int i = stateSize - shiftSize; i < stateSize - 1; ++i) {
            advance(i, i + 1, i + shiftSize - stateSize);
        }
        advance(stateSize - 1, 0, shiftSize - 1);
        next = 0;
    }

    void jump(uint64_t offset) {
        // Compute x^offset modulo the characteristic polynomial. Squaring
        // over GF(2) just doubles each power, before modular reduction.
        using Polynomial = std::bitset<2 * degree>;
        int bit = std::bit_width(offset) - 1;
        while (bit > 0 && (offset >> (bit - 1)) < degree) --bit;
        Polynomial remainder;
        remainder.set(offset >> bit);
        while (bit > 0) {
            int lowBit = (offset >> --bit) & 1;
            Polynomial squared;
            for (int i = 0; i < degree; ++i) {
                if (remainder[i]) squared.set(2 * i + lowBit);
            }
            for (int i = 2 * degree - 1; i >= degree; --i) {
                if (!squared[i]) continue;
                for (int power : polynomialPowers) squared.flip(i - degree + power);
            }
            remainder = squared;
        }

        // Evaluate the remainder at the transition by XORing the selected
        // 624-word windows into the resulting state.
        std::array<uint32_t, stateSize> jumped{};
        for (int i = 0; i < degree; ++i) {
            int head = i % stateSize;
            if (remainder[i]) {
                for (int j = 0; j < stateSize - head; ++j) jumped[j] ^= words[head + j];
                for (int j = 0; j < head; ++j) jumped[stateSize - head + j] ^= words[j];
            }
            advance(head, (head + 1) % stateSize, (head + shiftSize) % stateSize);
        }
        words = jumped;
    }

public:
    using result_type = uint32_t;

    explicit Mt19937(uint64_t offset) {
        words[0] = 42;
        for (int i = 1; i < stateSize; ++i) {
            words[i] = 1812433253u * (words[i - 1] ^ (words[i - 1] >> 30)) + i;
        }
        twist();
        // A few million draws are cheaper to skip by twisting whole blocks.
        if (offset < (uint64_t{1} << 24)) {
            for (; offset >= stateSize; offset -= stateSize) twist();
            next = static_cast<int>(offset);
        } else {
            jump(offset);
        }
    }

    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT32_MAX; }

    result_type operator()() {
        if (next == stateSize) twist();
        uint32_t value = words[next++];
        value ^= value >> 11;
        value ^= (value << 7) & 0x9d2c5680u;
        value ^= (value << 15) & 0xefc60000u;
        return value ^ (value >> 18);
    }
};

class RandomGenerator {
    Mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint64_t offset) : rng(offset), dist(0.0f, 1.0f) {}
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

    return octree.applyToTris(RaySegment(from, dir), [&](size_t idx, const Triangle& tri) {
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
// Distributed Simulation State
// ============================================================================

struct RowPartition {
    std::vector<int> counts;
    std::vector<int> offsets;
    int rank;

    RowPartition(int rows, int rank, int ranks) : counts(ranks), offsets(ranks), rank(rank) {
        int offset = 0;
        for (int r = 0; r < ranks; ++r) {
            counts[r] = rows / ranks + (r < rows % ranks);
            offsets[r] = offset;
            offset += counts[r];
        }
    }

    size_t begin() const { return offsets[rank]; }
    size_t size() const { return counts[rank]; }
    bool isRoot() const { return rank == 0; }
};

struct SimulationState {
    std::vector<Triangle> triangles;
    Octree octree;
    RowPartition rows;

    size_t numTriangles() const { return triangles.size(); }
    size_t numTimesteps;
    size_t sourceIndex;
    val_t reflectivity;

    // The pair matrices contain only the receiver rows assigned to this rank.
    std::unique_ptr<uint8_t[]> delays;
    std::unique_ptr<val_t[]> weights;
    int minDelay = MAX_DELAY;

    // Cross-correlation needs the complete local and source histories.
    std::unique_ptr<val_t[]> localRadiosity;
    std::unique_ptr<val_t[]> sourceRadiosity;
    uint64_t nonZeroKij = 0;

    SimulationState(int subdivisions, size_t timesteps, size_t sourceIdx,
                    val_t reflectivity, int rank, int ranks)
        : triangles(IcosphereMesh(subdivisions, ROOM_RADIUS).triangles),
          rows(static_cast<int>(triangles.size()), rank, ranks),
          numTimesteps(timesteps), sourceIndex(sourceIdx % triangles.size()),
          reflectivity(reflectivity) {
        if (rows.isRoot()) {
            printf("Generated icosphere mesh with %zu triangles\n", numTriangles());
            printf("Building octree...\n\n");
        }
        if (rows.size() != 0) octree.build(triangles);
    }
};

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state, bool validate) {
    const size_t n = state.numTriangles();
    const auto& rows = state.rows;
    if (rows.isRoot()) printf("Computing time delays (Tau)...\n");

    std::vector<Vec3> centers(n);
    std::vector<val_t> areas(n);
    for (size_t j = 0; j < n; ++j) {
        centers[j] = state.triangles[j].center();
        areas[j] = state.triangles[j].area();
    }

    const size_t pairs = rows.size() * n;
    state.delays = std::make_unique_for_overwrite<uint8_t[]>(pairs);
    state.weights = std::make_unique_for_overwrite<val_t[]>(pairs);
    uint64_t sampledPairs = 0;
    for (size_t local = 0; local < rows.size(); ++local) {
        const size_t i = rows.begin() + local;
        const Vec3 normal = state.triangles[i].normal();
        for (size_t j = 0; j < n; ++j) {
            uint8_t delay = 0;
            // Culled pairs consume no random draws and never contribute.
            if (i != j && normal.dot(state.triangles[j].normal()) <= 0.99f) {
                val_t dist = (centers[i] - centers[j]).norm();
                delay = static_cast<uint8_t>(std::ceil(dist * INV_WAVE_SPEED));
                ++sampledPairs;
            }
            state.delays[local * n + j] = delay;
        }
    }

    // The serial traversal uses four draws per ray for each sampled pair.
    // The prefix sum locates the first draw for these contiguous receiver rows.
    uint64_t precedingPairs = 0;
    MPI_Exscan(&sampledPairs, &precedingPairs, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (rows.isRoot()) {
        precedingPairs = 0;
        printf("Computing form factors (Kij)...\n");
    }

    int localMinDelay = MAX_DELAY;
    if (rows.size() != 0) {
        RandomGenerator rng(precedingPairs * (4 * NUM_RAYS));
        uint64_t nonZero = 0;
        for (size_t local = 0; local < rows.size(); ++local) {
            const size_t i = rows.begin() + local;
            for (size_t j = 0; j < n; ++j) {
                const size_t pair = local * n + j;
                const int delay = state.delays[pair];
                val_t weight = ZERO;
                if (delay != 0) {
                    const val_t kij = computeKij(i, j, state.triangles, state.octree, rng);
                    weight = kij * areas[j];
                    if (weight > ZERO) localMinDelay = std::min(localMinDelay, delay);
                    if (validate && kij > EPSILON) ++nonZero;
                }
                state.weights[pair] = weight;
            }
        }
        state.nonZeroKij = nonZero;
    }

    MPI_Allreduce(&localMinDelay, &state.minDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    if (rows.isRoot()) printf("  Progress: %zu/%zu triangles\n", n, n);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

// Clamp contributions and accumulate them in the serial emitter order.
val_t sumReflections(const val_t* weights, const uint8_t* delays, size_t n,
                     const std::array<const val_t*, MAX_DELAY + 1>& past) {
    val_t sum = ZERO;
    for (size_t j = 0; j < n; ++j) {
        if (weights[j] <= ZERO) continue;
        const val_t radJ = past[delays[j]][j];
        if (radJ <= ZERO) continue;
        sum += std::min(weights[j], ONE) * radJ;
    }
    return sum;
}

size_t runSimulation(SimulationState& state) {
    const auto& rows = state.rows;
    const size_t n = state.numTriangles();
    const size_t timesteps = state.numTimesteps;
    if (rows.isRoot()) printf("Running wave propagation simulation...\n");

    state.localRadiosity = std::make_unique_for_overwrite<val_t[]>(timesteps * rows.size());
    state.sourceRadiosity = std::make_unique_for_overwrite<val_t[]>(timesteps);
    if (timesteps == 0) return 0;

    // Every positive connection reads a previous timestep. Steps within a
    // batch have no dependencies on one another if the batch fits in minDelay.
    // Keep only the history window that future propagation can read.
    std::vector<val_t> history(MAX_DELAY * n, ZERO);
    const size_t batchSize = std::min<size_t>(state.minDelay, timesteps);
    auto batch = std::make_unique_for_overwrite<val_t[]>(batchSize * n);
    std::vector<int> counts(rows.counts.size());
    std::vector<int> offsets(rows.offsets.size());
    size_t batchLength = 0;

    for (size_t first = 0; first < timesteps; first += batchSize) {
        const size_t length = std::min(batchSize, timesteps - first);
        if (length != batchLength) {
            batchLength = length;
            for (size_t rank = 0; rank < counts.size(); ++rank) {
                counts[rank] = rows.counts[rank] * length;
                offsets[rank] = rows.offsets[rank] * length;
            }
        }

        // Store each rank's batch contiguously for an in-place allgather.
        val_t* const localBatch = batch.get() + offsets[rows.rank];
        for (size_t step = 0; step < length; ++step) {
            const size_t t = first + step;
            const bool hasArrivals = t >= static_cast<size_t>(state.minDelay);
            std::array<const val_t*, MAX_DELAY + 1> past{};
            for (int delay = state.minDelay; delay <= MAX_DELAY; ++delay) {
                past[delay] = history.data() + ((t + MAX_DELAY - delay) % MAX_DELAY) * n;
            }

            for (size_t local = 0; local < rows.size(); ++local) {
                const val_t sumB = hasArrivals ?
                    sumReflections(state.weights.get() + local * n,
                                   state.delays.get() + local * n, n, past) : ZERO;
                const val_t emission = rows.begin() + local == state.sourceIndex &&
                                       t < timesteps / 2 ? ONE : ZERO;
                localBatch[step * rows.size() + local] = state.reflectivity * sumB + emission;
            }
        }

        std::copy_n(localBatch, length * rows.size(),
                    state.localRadiosity.get() + first * rows.size());
        MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, batch.get(), counts.data(),
                       offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);

        for (size_t step = 0; step < length; ++step) {
            const size_t t = first + step;
            val_t* current = history.data() + (t % MAX_DELAY) * n;
            for (size_t rank = 0; rank < counts.size(); ++rank) {
                const size_t rankRows = rows.counts[rank];
                if (rankRows == 0) continue;
                std::copy_n(batch.get() + offsets[rank] + step * rankRows, rankRows,
                            current + rows.offsets[rank]);
            }
            state.sourceRadiosity[t] = current[state.sourceIndex];

            if (rows.isRoot() && ((t + 1) % 10 == 0 || t + 1 == timesteps)) {
                printf("  Timestep %zu/%zu\n", t + 1, timesteps);
            }
        }
    }

    return (MAX_DELAY + batchSize) * n * sizeof(val_t);
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

std::vector<val_t> computeDistances(const SimulationState& state) {
    const auto& rows = state.rows;
    if (rows.isRoot()) printf("Computing distances via cross-correlation...\n");
    std::vector<val_t> localDistances(rows.size());
    const size_t timesteps = state.numTimesteps;
    std::vector<val_t> trace(timesteps);

    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t t = 0; t < timesteps; ++t) {
            trace[t] = state.localRadiosity[t * rows.size() + i];
        }

        val_t maxCorr = ZERO;
        size_t bestT = 0;
        for (size_t t = 0; t < timesteps; ++t) {
            val_t sum = ZERO;
            for (size_t tt = t; tt < timesteps; ++tt) {
                sum += state.sourceRadiosity[tt - t] * trace[tt];
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = t;
            }
        }
        localDistances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    std::vector<val_t> distances(rows.isRoot() ? state.numTriangles() : 0);
    MPI_Gatherv(localDistances.data(), rows.counts[rows.rank], MPI_FLOAT, distances.data(),
                rows.counts.data(), rows.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    return distances;
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const std::vector<val_t>& distances, size_t sourceIndex,
                     uint64_t receivedEnergy, uint64_t nonZeroKij) {
    const size_t n = distances.size();
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < n; ++i) {
        val_t d = distances[i];
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(n));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, n);

    // Check source distance is zero or very small
    val_t srcDist = distances[sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    // Check radiosity propagation (some triangles should have received energy)
    printf("  Triangles receiving energy: %" PRIu64 "/%zu\n", receivedEnergy, n);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    printf("  Non-zero form factors: %" PRIu64 "/%zu (%.2f%%)\n",
           nonZeroKij, n * n,
           100.0f * nonZeroKij / static_cast<val_t>(n * n));

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

bool validateDistributedResults(const SimulationState& state, const std::vector<val_t>& distances) {
    const auto& rows = state.rows;
    uint64_t receivedEnergy = 0;
    for (size_t i = 0; i < rows.size(); ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.localRadiosity[t * rows.size() + i] > EPSILON) {
                ++receivedEnergy;
                break;
            }
        }
    }
    const uint64_t localCounts[] = {receivedEnergy, state.nonZeroKij};
    uint64_t globalCounts[2];
    MPI_Reduce(localCounts, globalCounts, 2, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);

    int passed = 0;
    if (rows.isRoot()) {
        passed = validateResults(distances, state.sourceIndex, globalCounts[0], globalCounts[1]);
    }
    MPI_Bcast(&passed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return passed != 0;
}

// ============================================================================
// Hash for Verification
// ============================================================================

uint64_t computeHash(const std::vector<val_t>& distances) {
    uint64_t hash = 0;
    for (size_t i = 0; i < distances.size(); ++i) {
        const uint32_t bits = std::bit_cast<uint32_t>(distances[i]);
        hash ^= (static_cast<uint64_t>(bits) + i) * 0x9e3779b97f4a7c15ULL;
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
    MPI_Init(&argc, &argv);
    struct FinalizeMPI {
        ~FinalizeMPI() { MPI_Finalize(); }
    } finalizeMPI;

    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

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
            if (rank == 0) printUsage(argv[0]);
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (timesteps < 0) {
        if (rank == 0) fprintf(stderr, "Timesteps must be non-negative.\n");
        return 1;
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    if (rank == 0) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    SimulationState state(subdivisions, static_cast<size_t>(timesteps),
                          static_cast<size_t>(sourceIdx), reflectivity, rank, ranks);
    using Clock = std::chrono::steady_clock;
    using Milliseconds = std::chrono::milliseconds;

    MPI_Barrier(MPI_COMM_WORLD);
    auto startPre = Clock::now();
    computeFormFactors(state, validate);
    long preDuration = std::chrono::duration_cast<Milliseconds>(Clock::now() - startPre).count();
    if (rank == 0) printf("Precomputation time: %ld ms\n\n", preDuration);

    auto startSim = Clock::now();
    size_t propagationMemory = runSimulation(state);
    MPI_Barrier(MPI_COMM_WORLD);
    long simDuration = std::chrono::duration_cast<Milliseconds>(Clock::now() - startSim).count();
    if (rank == 0) printf("Simulation time: %ld ms\n\n", simDuration);

    auto startDist = Clock::now();
    auto distances = computeDistances(state);
    long distDuration = std::chrono::duration_cast<Milliseconds>(Clock::now() - startDist).count();

    if (rank == 0) {
        printf("Distance computation time: %ld ms\n\n", distDuration);
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

        size_t n = state.numTriangles();
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n) * n;
        double simOps = kijOps * t;
        double distOps = static_cast<double>(n) * t * t;

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        const size_t pairMemory = n * n * (sizeof(val_t) + sizeof(uint8_t));
        const size_t traceMemory = t * (n + ranks) * sizeof(val_t);
        const size_t totalMem = pairMemory + traceMemory + ranks * propagationMemory;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));
        printf("  Result hash: %016" PRIX64 "\n\n", computeHash(distances));

        if (printResults) {
            std::vector<double> distData(distances.begin(), distances.end());
            print_results(distData, "Distances");
        }
    }

    return validate && !validateDistributedResults(state, distances) ? 1 : 0;
}
