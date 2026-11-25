/**
 * Room Response Simulation Benchmark
 * 
 * This is an MPI implementation of room impulse response simulation using
 * radiosity-based wave propagation. It models how sound/light
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

struct RaySegment {
    Vec3 halfDirection;
    Vec3 midpoint;
    Vec3 absHalfDirection;

    RaySegment(const Vec3& from, const Vec3& direction)
        : halfDirection(direction * 0.5f), midpoint(from + halfDirection),
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

// The std::mt19937 recurrence, with access to its untempered words for jumping
// ahead in the original serial random stream.
class Mt19937 {
    static constexpr size_t wordCount = 624;
    static constexpr size_t lag = 397;
    static constexpr size_t degree = 19937;
    std::array<uint32_t, wordCount> words;
    size_t next = wordCount;

    static uint32_t twist(uint32_t first, uint32_t second) {
        uint32_t joined = (first & 0x80000000U) | (second & 0x7fffffffU);
        return (joined >> 1) ^ ((0U - (second & 1U)) & 0x9908b0dfU);
    }

    void advanceBlock() {
        for (size_t i = 0; i < wordCount - lag; ++i) {
            words[i] = words[i + lag] ^ twist(words[i], words[i + 1]);
        }
        for (size_t i = wordCount - lag; i < wordCount - 1; ++i) {
            words[i] = words[i + lag - wordCount] ^ twist(words[i], words[i + 1]);
        }
        words.back() = words[lag - 1] ^ twist(words.back(), words.front());
    }

    uint32_t raw() {
        if (next == wordCount) {
            advanceBlock();
            next = 0;
        }
        return words[next++];
    }

public:
    using result_type = uint32_t;
    static constexpr result_type min() { return 0; }
    static constexpr result_type max() { return UINT32_MAX; }

    explicit Mt19937(uint32_t seed) {
        words[0] = seed;
        for (size_t i = 1; i < wordCount; ++i) {
            words[i] = 1812433253U * (words[i - 1] ^ (words[i - 1] >> 30))
                       + static_cast<uint32_t>(i);
        }
    }

    result_type operator()() {
        uint32_t word = raw();
        word ^= word >> 11;
        word ^= (word << 7) & 0x9d2c5680U;
        word ^= (word << 15) & 0xefc60000U;
        return word ^ (word >> 18);
    }

    // Called on a freshly seeded engine. The jump uses the characteristic
    // polynomial of the MT19937 transition over GF(2).
    void skip(uint64_t count) {
        if (count < (1ULL << 22)) {
            while (count >= wordCount) {
                advanceBlock();
                count -= wordCount;
            }
            if (count) {
                advanceBlock();
                next = count;
            }
            return;
        }

        // Non-leading exponents of the degree-19937 characteristic polynomial.
        constexpr int exponents[] = {
            0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227,
            3454, 3681, 3908, 4135, 4362, 4753, 5661, 6337, 6569, 7129,
            7477, 7525, 7583, 7752, 7979, 8206, 9505, 9901, 9969, 10128,
            10693, 10761, 10920, 11089, 11147, 11157, 11215, 11321, 11374, 11384,
            11485, 11611, 11712, 11717, 11838, 11881, 11944, 11997, 12277, 12335,
            12393, 12504, 12509, 12620, 12673, 12731, 12736, 12789, 12905, 12958,
            12963, 13137, 13185, 13190, 13243, 13301, 13412, 13528, 13533, 13639,
            13697, 13760, 13813, 13866, 14093, 14151, 14209, 14320, 14325, 14436,
            14547, 14552, 14605, 14721, 14774, 14779, 14953, 15001, 15006, 15059,
            15117, 15228, 15344, 15349, 15455, 15513, 15576, 15629, 15682, 15909,
            15967, 16025, 16136, 16141, 16252, 16363, 16368, 16421, 16537, 16590,
            16595, 16817, 16822, 16875, 16933, 17044, 17160, 17271, 17329, 17445,
            17498, 17725, 17783, 17841, 17952, 18068, 18179, 18237, 18406, 18633,
            18691, 18860, 19087, 19314
        };

        // Find x^count modulo the characteristic polynomial by squaring and
        // multiplying by x for each set exponent bit.
        std::array<uint64_t, (2 * degree + 63) / 64> poly{};
        poly[0] = 1;
        for (size_t bit = std::bit_width(count); bit-- > 0;) {
            decltype(poly) squared{};
            size_t shift = (count >> bit) & 1U;
            for (size_t i = 0; i < degree; ++i) {
                if ((poly[i / 64] >> (i % 64)) & 1U) {
                    size_t power = 2 * i + shift;
                    squared[power / 64] |= 1ULL << (power % 64);
                }
            }
            for (size_t i = 2 * degree - 1; i >= degree; --i) {
                if ((squared[i / 64] >> (i % 64)) & 1U) {
                    squared[i / 64] ^= 1ULL << (i % 64);
                    for (int exponent : exponents) {
                        size_t power = i - degree + exponent;
                        squared[power / 64] ^= 1ULL << (power % 64);
                    }
                }
            }
            poly = squared;
        }

        // Apply the polynomial to the first block of output words. Each
        // coefficient selects a shifted copy of that block to XOR into it.
        std::array<uint32_t, degree + wordCount - 1> sequence;
        for (auto& word : sequence) word = raw();
        words.fill(0);
        for (size_t i = 0; i < degree; ++i) {
            if ((poly[i / 64] >> (i % 64)) & 1U) {
                for (size_t j = 0; j < wordCount; ++j) words[j] ^= sequence[i + j];
            }
        }
        next = 0;
    }
};

class RandomGenerator {
    Mt19937 rng{42};
    std::uniform_real_distribution<val_t> dist{0.0f, 1.0f};
public:
    void skipPairs(uint64_t pairs) {
        // Each ray takes two samples per point and samples two points.
        // Each float sample consumes one MT word.
        rng.skip(pairs * NUM_RAYS * 4);
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

// Cull triangles facing the same direction before consuming any samples.
bool samplesPair(const Triangle& receiver, const Triangle& emitter) {
    return !(receiver.normal().dot(emitter.normal()) > 0.99f);
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

constexpr val_t ROOM_RADIUS = 10.0f;
constexpr int MAX_DELAY = static_cast<int>(2 * ROOM_RADIUS * INV_WAVE_SPEED);

struct RowPartition {
    MPI_Comm comm;
    int rank;
    std::vector<int> counts;
    std::vector<int> offsets;
    size_t first;
    size_t count;

    RowPartition(MPI_Comm comm, size_t numTriangles) : comm(comm) {
        int ranks;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        counts.resize(ranks);
        offsets.resize(ranks);
        int offset = 0;
        for (int r = 0; r < ranks; ++r) {
            offsets[r] = offset;
            counts[r] = numTriangles / ranks + (static_cast<size_t>(r) < numTriangles % ranks);
            offset += counts[r];
        }
        first = offsets[rank];
        count = counts[rank];
    }

    bool isRoot() const { return rank == 0; }
};

struct Interaction {
    // Even the largest supported mesh has fewer than 2^24 triangles. Keep
    // each emitter index, delay, and weight in eight bytes on the hot path.
    uint32_t sourceAndDelay;
    val_t weight;

    Interaction(size_t source, int delay, val_t weight)
        : sourceAndDelay((static_cast<uint32_t>(delay) << 24) | static_cast<uint32_t>(source)),
          weight(weight) {}

    int delay() const { return sourceAndDelay >> 24; }
    size_t source() const { return sourceAndDelay & 0x00ffffffU; }
};

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;
    size_t sourceIndex;
    val_t reflectivity;
    RowPartition rows;

    std::vector<size_t> rowOffsets;
    std::vector<Interaction> interactions;
    std::vector<val_t> history;     // Local triangle-major radiosity for correlation
    std::vector<val_t> distances;
    uint64_t nonZeroKij = 0;
    int minDelay = MAX_DELAY + 1;
    int maxDelay = 0;
    size_t propagationBufferBytes = 0;

    SimulationState(MPI_Comm comm, size_t n, size_t timesteps, size_t source, val_t rho)
        : numTriangles(n), numTimesteps(timesteps), sourceIndex(source % n), reflectivity(rho),
          rows(comm, n), rowOffsets(rows.count + 1), history(rows.count * timesteps),
          distances(rows.count) {}
};

// ============================================================================
// Local Form Factors and Time Delays
// ============================================================================

void precompute(SimulationState& state, int subdivisions) {
    const auto& rows = state.rows;
    const size_t n = state.numTriangles;
    if (rows.isRoot()) {
        printf("Generated icosphere mesh with %zu triangles\n", n);
        printf("Building octree...\n");
    }

    IcosphereMesh mesh(subdivisions, ROOM_RADIUS);
    const auto& triangles = mesh.triangles;
    Octree octree;
    octree.build(triangles);

    std::vector<Vec3> centers(n);
    std::vector<val_t> areas(n);
    for (size_t i = 0; i < n; ++i) {
        centers[i] = triangles[i].center();
        areas[i] = triangles[i].area();
    }

    if (rows.isRoot()) {
        printf("Computing time delays (Tau) and form factors (Kij)...\n");
    }

    RandomGenerator rng;
    if (rows.counts.size() > 1) {
        // Only non-culled off-diagonal pairs consume random numbers. Prefix
        // counts locate each rank's first sample in the serial row-major stream.
        uint64_t pairs = 0;
        for (size_t i = rows.first; i < rows.first + rows.count; ++i) {
            for (size_t j = 0; j < n; ++j) {
                pairs += i != j && samplesPair(triangles[i], triangles[j]);
            }
        }
        uint64_t precedingPairs = 0;
        MPI_Exscan(&pairs, &precedingPairs, 1, MPI_UINT64_T, MPI_SUM, rows.comm);
        if (!rows.isRoot()) rng.skipPairs(precedingPairs);
    }

    if (state.numTimesteps > MAX_DELAY) state.interactions.reserve(rows.count * n);
    for (size_t local = 0; local < rows.count; ++local) {
        size_t i = rows.first + local;
        for (size_t j = 0; j < n; ++j) {
            if (i == j || !samplesPair(triangles[i], triangles[j])) continue;
            val_t kij = computeKij(i, j, triangles, octree, rng);
            state.nonZeroKij += kij > EPSILON;
            if (kij <= ZERO) continue;

            int delay = static_cast<int>(std::ceil((centers[i] - centers[j]).norm()
                                                   * INV_WAVE_SPEED));
            if (static_cast<size_t>(delay) >= state.numTimesteps) continue;
            state.interactions.emplace_back(j, delay, kij * areas[j]);
            state.minDelay = std::min(state.minDelay, delay);
            state.maxDelay = std::max(state.maxDelay, delay);
        }
        state.rowOffsets[local + 1] = state.interactions.size();
    }

    int limits[] = {state.minDelay, -state.maxDelay};
    MPI_Allreduce(MPI_IN_PLACE, limits, 2, MPI_INT, MPI_MIN, rows.comm);
    state.minDelay = limits[0];
    state.maxDelay = -limits[1];
    if (rows.isRoot()) printf("  Progress: %zu/%zu triangles\n", n, n);
}

// ============================================================================
// Wave Propagation
// ============================================================================

void runSimulation(SimulationState& state) {
    const auto& rows = state.rows;
    const size_t n = state.numTriangles;
    const size_t timesteps = state.numTimesteps;
    if (rows.isRoot()) printf("Running wave propagation simulation...\n");
    if (timesteps == 0) return;

    // Distinct triangle centers have positive delays. No value produced in
    // a batch can be needed until a later batch.
    const size_t batchSize = state.maxDelay ? state.minDelay : timesteps;
    const bool exchange = rows.counts.size() > 1 && batchSize < timesteps;
    const size_t ringRows = state.maxDelay ? state.maxDelay + batchSize : 0;
    std::vector<val_t> ring(ringRows * n);
    std::vector<val_t> batch(exchange ? batchSize * n : 0);
    std::vector<val_t> zeros(state.maxDelay ? n : 0);
    std::vector<int> counts, offsets;
    if (exchange) {
        counts.resize(rows.counts.size());
        offsets.resize(rows.counts.size());
        for (size_t r = 0; r < counts.size(); ++r) {
            counts[r] = batchSize * rows.counts[r];
            offsets[r] = batchSize * rows.offsets[r];
        }
    }
    state.propagationBufferBytes = (ring.size() + batch.size() + zeros.size()) * sizeof(val_t);

    std::array<const val_t*, MAX_DELAY + 1> previous;
    for (size_t begin = 0; begin < timesteps; begin += batchSize) {
        const size_t end = begin + std::min(batchSize, timesteps - begin);
        for (size_t t = begin; t < end; ++t) {
            for (int delay = state.minDelay; delay <= state.maxDelay; ++delay) {
                previous[delay] = static_cast<size_t>(delay) > t
                    ? zeros.data() : ring.data() + ((t - delay) % ringRows) * n;
            }

            val_t* output = nullptr;
            if (end < timesteps) {
                output = exchange ? batch.data() + offsets[rows.rank] + (t - begin) * rows.count
                                  : ring.data() + (t % ringRows) * n;
            }
            for (size_t i = 0; i < rows.count; ++i) {
                val_t sum = ZERO;
                for (size_t k = state.rowOffsets[i]; k < state.rowOffsets[i + 1]; ++k) {
                    const auto& link = state.interactions[k];
                    val_t radJ = previous[link.delay()][link.source()];
                    if (radJ <= ZERO) continue;
                    sum += std::min(link.weight, ONE) * radJ;
                }
                val_t emission = rows.first + i == state.sourceIndex && t < timesteps / 2
                    ? ONE : ZERO;
                val_t radB = state.reflectivity * sum + emission;
                state.history[i * timesteps + t] = radB;
                if (output) output[i] = radB;
            }
            if (rows.isRoot() && ((t + 1) % 10 == 0 || t + 1 == timesteps)) {
                printf("  Timestep %zu/%zu\n", t + 1, timesteps);
            }
        }

        if (exchange && end < timesteps) {
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, batch.data(), counts.data(),
                           offsets.data(), MPI_FLOAT, rows.comm);
            for (size_t t = begin; t < end; ++t) {
                val_t* slot = ring.data() + (t % ringRows) * n;
                for (size_t r = 0; r < counts.size(); ++r) {
                    const val_t* source = batch.data() + offsets[r] + (t - begin) * rows.counts[r];
                    std::copy_n(source, rows.counts[r], slot + rows.offsets[r]);
                }
            }
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

#if defined(__GNUC__) && !defined(__clang__)
// GCC rounds products separately in the original strided correlation loop.
// Retain that rounding when the contiguous histories enable vectorization.
#pragma GCC push_options
#pragma GCC optimize ("fp-contract=off")
#endif
void computeDistances(SimulationState& state) {
    const auto& rows = state.rows;
    const size_t timesteps = state.numTimesteps;
    if (rows.isRoot()) printf("Computing distances via cross-correlation...\n");
    if (timesteps == 0) return;

    const int owner = std::upper_bound(rows.offsets.begin(), rows.offsets.end(), state.sourceIndex)
        - rows.offsets.begin() - 1;
    std::vector<val_t> sourceHistory(rows.counts.size() > 1 ? timesteps : 0);
    const val_t* source;
    if (rows.counts.size() == 1) {
        source = state.history.data() + state.sourceIndex * timesteps;
    } else {
        if (owner == rows.rank) {
            size_t local = state.sourceIndex - rows.first;
            std::copy_n(state.history.data() + local * timesteps, timesteps, sourceHistory.data());
        }
        MPI_Bcast(sourceHistory.data(), static_cast<int>(timesteps), MPI_FLOAT, owner, rows.comm);
        source = sourceHistory.data();
    }

    for (size_t i = 0; i < rows.count; ++i) {
        const val_t* receiver = state.history.data() + i * timesteps;
        val_t maxCorr = ZERO;
        size_t bestT = 0;
        for (size_t t = 0; t < timesteps; ++t) {
            val_t sum = ZERO;
            for (size_t tt = t; tt < timesteps; ++tt) {
                sum += source[tt - t] * receiver[tt];
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = t;
            }
        }
        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC pop_options
#endif

// ============================================================================
// Validation and Result Hash
// ============================================================================

uint64_t countEnergy(const SimulationState& state) {
    uint64_t receivedEnergy = 0;
    for (size_t i = 0; i < state.rows.count; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.history[i * state.numTimesteps + t] > EPSILON) {
                ++receivedEnergy;
                break;
            }
        }
    }
    return receivedEnergy;
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.numTriangles);

    val_t srcDist = distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    printf("  Triangles receiving energy: %" PRIu64 "/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    const size_t pairs = state.numTriangles * state.numTriangles;
    printf("  Non-zero form factors: %" PRIu64 "/%zu (%.2f%%)\n", nonZeroKij, pairs,
           100.0f * nonZeroKij / static_cast<val_t>(pairs));

    if (nonZeroKij == 0) {
        printf("  ERROR: All form factors are zero - visibility computation failed\n");
        return false;
    }

    if (!allNonNegative) return false;

    printf("  Validation: PASSED\n");
    return true;
}

uint64_t computeHash(const SimulationState& state) {
    uint64_t hash = 0;
    for (size_t i = 0; i < state.rows.count; ++i) {
        uint64_t bits = std::bit_cast<uint32_t>(state.distances[i]);
        hash ^= (bits + state.rows.first + i) * 0x9e3779b97f4a7c15ULL;
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

int benchmark(int argc, char** argv, int rank, int ranks) {
    const bool root = rank == 0;
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
            if (root) printUsage(argv[0]);
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            return 1;
        }
    }

    if (timesteps < 0) {
        if (root) fprintf(stderr, "Number of timesteps must be non-negative\n");
        return 1;
    }

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    const size_t n = size_t{20} << (2 * subdivisions);
    const size_t t = timesteps;

    MPI_Comm comm = MPI_COMM_WORLD;
    if (static_cast<size_t>(ranks) > n) {
        MPI_Comm_split(MPI_COMM_WORLD, static_cast<size_t>(rank) < n ? 0 : MPI_UNDEFINED,
                       rank, &comm);
        if (comm == MPI_COMM_NULL) return 0;
    }

    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("\n");
    }

    SimulationState state(comm, n, t, static_cast<size_t>(sourceIdx), reflectivity);

    auto phaseTime = [&](double start, const char* label) {
        double elapsed = MPI_Wtime() - start;
        double maxElapsed = 0;
        MPI_Reduce(&elapsed, &maxElapsed, 1, MPI_DOUBLE, MPI_MAX, 0, comm);
        long milliseconds = static_cast<long>(maxElapsed * 1000);
        if (root) printf("%s time: %ld ms\n\n", label, milliseconds);
        return milliseconds;
    };

    double start = MPI_Wtime();
    precompute(state, subdivisions);
    long preDuration = phaseTime(start, "Precomputation");

    start = MPI_Wtime();
    runSimulation(state);
    long simDuration = phaseTime(start, "Simulation");

    start = MPI_Wtime();
    computeDistances(state);
    long distDuration = phaseTime(start, "Distance computation");

    uint64_t localHash = computeHash(state);
    uint64_t hash = 0;
    MPI_Reduce(&localHash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, comm);

    uint64_t localMemory = state.interactions.capacity() * sizeof(Interaction)
        + state.rowOffsets.size() * sizeof(size_t)
        + (state.history.size() + state.distances.size()) * sizeof(val_t)
        + state.propagationBufferBytes;
    uint64_t maxMemory = 0;
    MPI_Reduce(&localMemory, &maxMemory, 1, MPI_UINT64_T, MPI_MAX, 0, comm);

    std::vector<val_t> distances;
    if (validate || printResults) {
        if (root) distances.resize(n);
        MPI_Gatherv(state.distances.data(), static_cast<int>(state.rows.count), MPI_FLOAT,
                    distances.data(), state.rows.counts.data(), state.rows.offsets.data(),
                    MPI_FLOAT, 0, comm);
    }

    uint64_t validationCounts[2] = {};
    if (validate) {
        uint64_t localCounts[] = {countEnergy(state), state.nonZeroKij};
        MPI_Reduce(localCounts, validationCounts, 2, MPI_UINT64_T, MPI_SUM, 0, comm);
    }

    int result = 0;
    if (root) {
        const long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", static_cast<double>(n) * n);
        printf("  Simulation operations: %.2e\n", static_cast<double>(n) * n * t);
        printf("  Distance computations: %.2e\n", static_cast<double>(n) * t * t);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
        printf("  Memory usage: %.2f MB (per-rank simulation buffers, maximum)\n",
               maxMemory / (1024.0 * 1024.0));
        printf("  Result hash: %016" PRIX64 "\n\n", hash);

        if (printResults) {
            std::vector<double> distData(distances.begin(), distances.end());
            print_results(distData, "Distances");
        }
        if (validate && !validateResults(state, distances, validationCounts[0], validationCounts[1])) {
            result = 1;
        }
    }

    if (validate) MPI_Bcast(&result, 1, MPI_INT, 0, comm);
    if (comm != MPI_COMM_WORLD) MPI_Comm_free(&comm);
    return result;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    int result;
    try {
        result = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
        return 1;
    }
    MPI_Finalize();
    return result;
}
