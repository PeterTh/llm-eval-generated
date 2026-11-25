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
#include <cinttypes>
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
constexpr val_t ROOM_RADIUS = 10.0f;
// The extra step allows for rounding at the diameter of the room.
constexpr size_t MAX_TIME_DELAY = static_cast<size_t>(2 * ROOM_RADIUS * INV_WAVE_SPEED) + 1;
static_assert(MAX_TIME_DELAY <= std::numeric_limits<uint8_t>::max());
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

    RaySegment(const Vec3& from, const Vec3& to)
        : halfDirection((to - from) * 0.5f), midpoint(from + halfDirection),
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
        Vec3 c = ray.midpoint - center;
        const Vec3& ad = ray.absHalfDirection;

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
    bool applyToTris(const RaySegment& ray, const Func& func) const {
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

class RandomGenerator {
    // Store the next block of untempered MT19937 words so we can jump in
    // the same stream as std::mt19937 without replaying other ranks' draws.
    static constexpr size_t blockSize = std::mt19937::state_size;
    static constexpr size_t middle = std::mt19937::shift_size;
    std::array<uint32_t, blockSize> words;
    size_t next = blockSize;
    std::uniform_real_distribution<val_t> dist;

    static uint32_t twist(uint32_t first, uint32_t second, uint32_t shifted) {
        uint32_t joined = (first & 0x80000000u) | (second & 0x7fffffffu);
        return shifted ^ (joined >> 1) ^ ((0u - (joined & 1u)) & std::mt19937::xor_mask);
    }

    void refill() {
        for (size_t i = 0; i < blockSize - middle; ++i) {
            words[i] = twist(words[i], words[i + 1], words[i + middle]);
        }
        for (size_t i = blockSize - middle; i < blockSize - 1; ++i) {
            words[i] = twist(words[i], words[i + 1], words[i + middle - blockSize]);
        }
        words.back() = twist(words.back(), words.front(), words[middle - 1]);
        next = 0;
    }

    // Coefficients below the leading term of the characteristic polynomial
    // of the standard MT19937 single-word transition over GF(2). Applying
    // Berlekamp-Massey to 2 * degree output low bits gives the recurrence.
    static constexpr size_t degree = 19937;
    static constexpr int terms[] = {
        0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681,
        3908, 4135, 4362, 4753, 5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752,
        7979, 8206, 9505, 9901, 9969, 10128, 10693, 10761, 10920, 11089, 11147,
        11157, 11215, 11321, 11374, 11384, 11485, 11611, 11712, 11717, 11838,
        11881, 11944, 11997, 12277, 12335, 12393, 12504, 12509, 12620, 12673,
        12731, 12736, 12789, 12905, 12958, 12963, 13137, 13185, 13190, 13243,
        13301, 13412, 13528, 13533, 13639, 13697, 13760, 13813, 13866, 14093,
        14151, 14209, 14320, 14325, 14436, 14547, 14552, 14605, 14721, 14774,
        14779, 14953, 15001, 15006, 15059, 15117, 15228, 15344, 15349, 15455,
        15513, 15576, 15629, 15682, 15909, 15967, 16025, 16136, 16141, 16252,
        16363, 16368, 16421, 16537, 16590, 16595, 16817, 16822, 16875, 16933,
        17044, 17160, 17271, 17329, 17445, 17498, 17725, 17783, 17841, 17952,
        18068, 18179, 18237, 18406, 18633, 18691, 18860, 19087, 19314
    };
    static constexpr size_t polynomialWords = (degree + 63) / 64;

    static uint64_t squareBits(uint64_t bits) {
        bits = (bits | (bits << 16)) & 0x0000ffff0000ffffULL;
        bits = (bits | (bits << 8))  & 0x00ff00ff00ff00ffULL;
        bits = (bits | (bits << 4))  & 0x0f0f0f0f0f0f0f0fULL;
        bits = (bits | (bits << 2))  & 0x3333333333333333ULL;
        return (bits | (bits << 1)) & 0x5555555555555555ULL;
    }

    void jump(uint64_t draws) {
        // Compute x^draws modulo the characteristic polynomial by squaring
        // over GF(2). Multiplying by x is an extra shift after squaring.
        std::array<uint64_t, 2 * polynomialWords> polynomial{};
        polynomial[0] = 1;
        for (int bit = std::bit_width(draws) - 1; bit >= 0; --bit) {
            const unsigned shift = (draws >> bit) & 1u;
            for (size_t i = polynomialWords; i-- > 0;) {
                uint64_t word = polynomial[i];
                polynomial[2 * i] = squareBits(static_cast<uint32_t>(word)) << shift;
                polynomial[2 * i + 1] = squareBits(word >> 32) << shift;
            }
            // The gap below the leading coefficient exceeds 64, so a
            // whole word can be reduced before any lower word is visited.
            for (size_t i = polynomial.size(); i > polynomialWords - 1;) {
                --i;
                size_t first = 64 * i;
                uint64_t high = polynomial[i];
                if (i == polynomialWords - 1) {
                    high >>= degree % 64;
                    polynomial[i] ^= high << (degree % 64);
                    first = degree;
                }
                if (high == 0) continue;
                for (int term : terms) {
                    size_t target = first - degree + term;
                    unsigned offset = target % 64;
                    polynomial[target / 64] ^= high << offset;
                    if (offset != 0) polynomial[target / 64 + 1] ^= high >> (64 - offset);
                }
            }
        }

        // Evaluate the polynomial on the rolling untempered MT state.
        std::array<uint32_t, blockSize> result{};
        size_t head = 0;
        for (size_t i = 0; i < degree; ++i) {
            if ((polynomial[i / 64] >> (i % 64)) & 1u) {
                for (size_t j = 0; j < blockSize - head; ++j) {
                    result[j] ^= words[head + j];
                }
                for (size_t j = 0; j < head; ++j) {
                    result[blockSize - head + j] ^= words[j];
                }
            }
            size_t following = (head + 1) % blockSize;
            words[head] = twist(words[head], words[following], words[(head + middle) % blockSize]);
            head = following;
        }
        words = result;
    }

public:
    using result_type = std::mt19937::result_type;
    static constexpr result_type min() { return std::mt19937::min(); }
    static constexpr result_type max() { return std::mt19937::max(); }

    explicit RandomGenerator(uint64_t draws) : dist(0.0f, 1.0f) {
        words[0] = 42;
        for (size_t i = 1; i < blockSize; ++i) {
            uint32_t previous = words[i - 1];
            words[i] = std::mt19937::initialization_multiplier * (previous ^ (previous >> 30)) + i;
        }
        refill();
        if (draws >= (1u << 23)) {
            jump(draws);
        } else {
            for (; draws >= blockSize; draws -= blockSize) refill();
            next = draws;
        }
    }

    result_type operator()() {
        if (next == blockSize) refill();
        uint32_t value = words[next++];
        value ^= value >> 11;
        value ^= (value << 7) & std::mt19937::tempering_b;
        value ^= (value << 15) & std::mt19937::tempering_c;
        return value ^ (value >> 18);
    }

    val_t rand() { return dist(*this); }
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

    return octree.applyToTris(RaySegment(from, to), [&](size_t idx, const Triangle& tri) {
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

// ============================================================================
// Distributed Simulation State
// ============================================================================

struct RowPartition {
    MPI_Comm comm;
    int rank;
    std::vector<int> counts;
    std::vector<int> offsets;

    RowPartition(MPI_Comm communicator, int rank, int ranks, size_t totalRows)
        : comm(communicator), rank(rank), counts(ranks), offsets(ranks) {
        for (int r = 0; r < ranks; ++r) {
            counts[r] = totalRows / ranks + (static_cast<size_t>(r) < totalRows % ranks);
            if (r > 0) offsets[r] = offsets[r - 1] + counts[r - 1];
        }
    }

    bool isRoot() const { return rank == 0; }
    size_t first() const { return offsets[rank]; }
    size_t size() const { return counts[rank]; }
};

struct SimulationState {
    const RowPartition& rows;
    const size_t numTriangles;
    const size_t numTimesteps;
    const size_t sourceIndex;
    const val_t reflectivity;

    std::vector<Triangle> triangles;
    Octree octree;
    std::vector<val_t> areas;
    std::vector<val_t> weights;       // Local receiver rows, in global emitter order
    std::vector<uint8_t> delays;     // Zero identifies an inactive pair
    std::vector<val_t> radB;         // Local histories, in triangle-major order
    std::vector<val_t> sourceHistory;
    std::vector<val_t> distances;

    uint64_t nonZeroKij = 0;
    int minDelay = std::numeric_limits<int>::max();
    int maxDelay = 0;

    SimulationState(const RowPartition& rows, size_t triangles, size_t timesteps,
                    size_t source, val_t reflectivity)
        : rows(rows), numTriangles(triangles), numTimesteps(timesteps),
          sourceIndex(source % triangles), reflectivity(reflectivity) {}
};

void initializeSimulation(SimulationState& state, IcosphereMesh& mesh) {
    state.triangles = std::move(mesh.triangles);
    if (state.rows.isRoot()) {
        printf("Generated icosphere mesh with %zu triangles\n", state.numTriangles);
        printf("Building octree...\n");
    }
    state.octree.build(state.triangles);

    state.areas.resize(state.numTriangles);
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }
    state.weights.resize(state.rows.size() * state.numTriangles);
    state.delays.resize(state.weights.size());
    state.radB.resize(state.rows.size() * state.numTimesteps);
    state.sourceHistory.resize(state.numTimesteps);
    state.distances.resize(state.rows.size());
}

// ============================================================================
// Precomputation Phase
// ============================================================================

uint64_t computeTimeDelays(SimulationState& state) {
    if (state.rows.isRoot()) printf("Computing time delays (Tau)...\n");
    const size_t n = state.numTriangles;
    std::vector<Vec3> centers(n);
    for (size_t i = 0; i < n; ++i) centers[i] = state.triangles[i].center();

    uint64_t sampledPairs = 0;
    for (size_t local = 0; local < state.rows.size(); ++local) {
        size_t i = state.rows.first() + local;
        auto* delays = state.delays.data() + local * n;
        const Vec3 normal = state.triangles[i].normal();
        for (size_t j = 0; j < n; ++j) {
            if (i == j || normal.dot(state.triangles[j].normal()) > 0.99f) continue;

            // Distinct eligible centers on the room mesh yield positive delays.
            delays[j] = computeTau(centers[i], centers[j]);
            ++sampledPairs;
        }
    }
    return sampledPairs;
}

void computeFormFactors(SimulationState& state, uint64_t sampledPairs) {
    if (state.rows.isRoot()) printf("Computing form factors (Kij)...\n");
    uint64_t precedingPairs = 0;
    MPI_Exscan(&sampledPairs, &precedingPairs, 1, MPI_UINT64_T, MPI_SUM, state.rows.comm);
    if (state.rows.isRoot()) precedingPairs = 0;

    // Two barycentric coordinates for each of two points per ray. A float
    // uniform draw uses one 32-bit MT19937 word.
    RandomGenerator rng(precedingPairs * (4 * NUM_RAYS));
    const size_t n = state.numTriangles;
    for (size_t local = 0; local < state.rows.size(); ++local) {
        size_t i = state.rows.first() + local;
        auto* delays = state.delays.data() + local * n;
        auto* weights = state.weights.data() + local * n;
        for (size_t j = 0; j < n; ++j) {
            if (delays[j] == 0) continue;
            val_t kij = computeKij(i, j, state.triangles, state.octree, rng);
            if (kij > EPSILON) ++state.nonZeroKij;
            if (kij <= ZERO) {
                delays[j] = 0;
                continue;
            }
            weights[j] = std::min(kij * state.areas[j], ONE);
            state.minDelay = std::min(state.minDelay, static_cast<int>(delays[j]));
            state.maxDelay = std::max(state.maxDelay, static_cast<int>(delays[j]));
        }
    }

    int localDelays[] = {-state.minDelay, state.maxDelay};
    int globalDelays[2];
    MPI_Allreduce(localDelays, globalDelays, 2, MPI_INT, MPI_MAX, state.rows.comm);
    state.minDelay = -globalDelays[0];
    state.maxDelay = globalDelays[1];
    if (state.rows.isRoot()) printf("  Progress: %zu/%zu triangles\n", n, n);
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

size_t runSimulation(SimulationState& state) {
    if (state.rows.isRoot()) printf("Running wave propagation simulation...\n");
    const size_t n = state.numTriangles;
    const size_t timesteps = state.numTimesteps;
    if (timesteps == 0) return 0;

    // No row needs data from the current batch, since every contributing
    // delay is at least the batch width. Keep only batches still in range.
    const size_t width = std::min({static_cast<size_t>(state.minDelay), timesteps, MAX_TIME_DELAY});
    const size_t maxDelay = state.maxDelay;
    const size_t slots = 1 + std::min((maxDelay + width - 1) / width, (timesteps - 1) / width);
    std::vector<val_t> history(slots * n * width);
    std::vector<size_t> timeOffsets(width * (maxDelay + 1));
    std::vector<int> counts = state.rows.counts;
    std::vector<int> offsets = state.rows.offsets;
    for (int& count : counts) count *= width;
    for (int& offset : offsets) offset *= width;
    const bool uniformCounts = counts.front() == counts.back();
    const bool distributed = counts.size() > 1;

    for (size_t begin = 0; begin < timesteps; begin += width) {
        const size_t steps = std::min(width, timesteps - begin);
        auto* current = history.data() + (begin / width % slots) * n * width;
        for (size_t step = 0; step < steps; ++step) {
            size_t t = begin + step;
            auto* offsetsAtStep = timeOffsets.data() + step * (maxDelay + 1);
            for (size_t delay = 1; delay <= std::min(t, maxDelay); ++delay) {
                size_t sourceTime = t - delay;
                offsetsAtStep[delay] = (sourceTime / width % slots) * n * width + sourceTime % width;
            }
        }

        for (size_t local = 0; local < state.rows.size(); ++local) {
            size_t i = state.rows.first() + local;
            const auto* weights = state.weights.data() + local * n;
            const auto* delays = state.delays.data() + local * n;
            for (size_t step = 0; step < steps; ++step) {
                size_t t = begin + step;
                const auto* offsetsAtStep = timeOffsets.data() + step * (maxDelay + 1);
                val_t sumB = ZERO;
                if (t >= width) {
                    for (size_t j = 0; j < n; ++j) {
                        unsigned delay = delays[j];
                        if (delay == 0 || delay > t) continue;
                        val_t radJ = history[offsetsAtStep[delay] + j * width];
                        if (radJ <= ZERO) continue;
                        sumB += weights[j] * radJ;
                    }
                }

                val_t emission = (i == state.sourceIndex && t < timesteps / 2) ? ONE : ZERO;
                val_t reflected = state.reflectivity * sumB + emission;
                state.radB[local * timesteps + t] = reflected;
                current[i * width + step] = reflected;
            }
        }

        if (distributed) {
            if (uniformCounts) {
                MPI_Allgather(MPI_IN_PLACE, 0, MPI_FLOAT, current, counts.front(), MPI_FLOAT,
                              state.rows.comm);
            } else {
                MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, current, counts.data(), offsets.data(),
                               MPI_FLOAT, state.rows.comm);
            }
        }
        std::copy_n(current + state.sourceIndex * width, steps, state.sourceHistory.data() + begin);

        if (state.rows.isRoot()) {
            for (size_t step = 1; step <= steps; ++step) {
                size_t completed = begin + step;
                if (completed % 10 == 0 || completed == timesteps) {
                    printf("  Timestep %zu/%zu\n", completed, timesteps);
                }
            }
        }
    }
    return history.size();
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    if (state.rows.isRoot()) printf("Computing distances via cross-correlation...\n");

    const size_t timesteps = state.numTimesteps;
    if (timesteps == 0) return;
    for (size_t i = 0; i < state.rows.size(); ++i) {
        const auto* history = state.radB.data() + i * timesteps;
        val_t maxCorr = ZERO;
        size_t bestT = 0;
        for (size_t t = 0; t < timesteps; ++t) {
            val_t sum = ZERO;
            for (size_t tt = t; tt < timesteps; ++tt) {
                sum += state.sourceHistory[tt - t] * history[tt];
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = t;
            }
        }
        state.distances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(const SimulationState& state, const std::vector<val_t>& distances,
                     uint64_t receivedEnergy, uint64_t nonZeroKij) {
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.numTriangles; ++i) {
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

    // Check source distance is zero or very small
    val_t srcDist = distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    printf("  Triangles receiving energy: %" PRIu64 "/%zu\n", receivedEnergy, state.numTriangles);
    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    const uint64_t totalPairs = uint64_t{state.numTriangles} * state.numTriangles;
    printf("  Non-zero form factors: %" PRIu64 "/%" PRIu64 " (%.2f%%)\n",
           nonZeroKij, totalPairs, 100.0f * nonZeroKij / static_cast<val_t>(totalPairs));

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
    uint64_t localHash = 0;
    for (size_t i = 0; i < state.distances.size(); ++i) {
        uint64_t bits = std::bit_cast<uint32_t>(state.distances[i]);
        localHash ^= (bits + state.rows.first() + i) * 0x9e3779b97f4a7c15ULL;
    }
    uint64_t hash = 0;
    MPI_Reduce(&localHash, &hash, 1, MPI_UINT64_T, MPI_BXOR, 0, state.rows.comm);
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

long phaseMilliseconds(double start, const RowPartition& rows) {
    double elapsed = MPI_Wtime() - start;
    double longest = 0.0;
    MPI_Reduce(&elapsed, &longest, 1, MPI_DOUBLE, MPI_MAX, 0, rows.comm);
    return static_cast<long>(1000.0 * longest);
}

int runBenchmark(const RowPartition& rows, int subdivisions, int targetTriangles, int timesteps,
                 int sourceIdx, val_t reflectivity, bool validate, bool printResults) {
    if (rows.isRoot()) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    IcosphereMesh mesh(subdivisions, ROOM_RADIUS);
    SimulationState state(rows, mesh.triangles.size(), static_cast<size_t>(timesteps),
                          static_cast<size_t>(sourceIdx), reflectivity);
    initializeSimulation(state, mesh);
    if (rows.isRoot()) printf("\n");

    MPI_Barrier(rows.comm);
    double start = MPI_Wtime();
    computeFormFactors(state, computeTimeDelays(state));
    long preDuration = phaseMilliseconds(start, rows);
    if (rows.isRoot()) printf("Precomputation time: %ld ms\n\n", preDuration);

    start = MPI_Wtime();
    size_t historyElements = runSimulation(state);
    long simDuration = phaseMilliseconds(start, rows);
    if (rows.isRoot()) printf("Simulation time: %ld ms\n\n", simDuration);

    start = MPI_Wtime();
    computeDistances(state);
    long distDuration = phaseMilliseconds(start, rows);
    if (rows.isRoot()) printf("Distance computation time: %ld ms\n\n", distDuration);

    uint64_t hash = computeHash(state);
    const size_t n = state.numTriangles;
    const size_t t = state.numTimesteps;
    uint64_t localMemory = state.weights.size() * sizeof(val_t) +
                           state.delays.size() * sizeof(uint8_t) +
                           (state.radB.size() + state.sourceHistory.size() + historyElements) * sizeof(val_t);
    uint64_t totalMem = 0;
    MPI_Reduce(&localMemory, &totalMem, 1, MPI_UINT64_T, MPI_SUM, 0, rows.comm);
    if (rows.isRoot()) {
        long totalTime = preDuration + simDuration + distDuration;
        const double kijOps = static_cast<double>(n) * n;
        printf("Total computation time: %ld ms\n", totalTime);
        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", kijOps * t);
        printf("  Distance computations: %.2e\n", static_cast<double>(n) * t * t);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));
        printf("  Result hash: %016" PRIX64 "\n\n", hash);
    }

    std::vector<val_t> distances;
    if (printResults || validate) {
        if (rows.isRoot()) distances.resize(n);
        MPI_Gatherv(state.distances.data(), rows.counts[rows.rank], MPI_FLOAT, distances.data(),
                    rows.counts.data(), rows.offsets.data(), MPI_FLOAT, 0, rows.comm);
    }
    if (printResults && rows.isRoot()) {
        std::vector<double> distData(distances.begin(), distances.end());
        print_results(distData, "Distances");
    }

    int status = 0;
    if (validate) {
        uint64_t localCounts[] = {0, state.nonZeroKij};
        auto history = state.radB.begin();
        for (size_t i = 0; i < rows.size(); ++i, history += t) {
            localCounts[0] += std::any_of(history, history + t, [](val_t value) {
                return value > EPSILON;
            });
        }
        uint64_t counts[2];
        MPI_Reduce(localCounts, counts, 2, MPI_UINT64_T, MPI_SUM, 0, rows.comm);
        if (rows.isRoot() && !validateResults(state, distances, counts[0], counts[1])) status = 1;
    }
    return status;
}

int benchmarkMain(int argc, char** argv, int rank, int ranks) {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

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
        if (rank == 0) fprintf(stderr, "Timesteps must be non-negative\n");
        return 1;
    }

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    const int triangles = 20 << (2 * subdivisions);
    MPI_Comm comm = MPI_COMM_WORLD;
    if (ranks > triangles) {
        MPI_Comm_split(MPI_COMM_WORLD, rank < triangles ? 0 : MPI_UNDEFINED, rank, &comm);
    }

    int status = 0;
    if (comm != MPI_COMM_NULL) {
        const RowPartition rows(comm, rank, std::min(ranks, triangles), triangles);
        status = runBenchmark(rows, subdivisions, targetTriangles, timesteps, sourceIdx,
                              reflectivity, validate, printResults);
        if (comm != MPI_COMM_WORLD) MPI_Comm_free(&comm);
    }
    if (validate) MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return status;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);

    int status;
    try {
        status = benchmarkMain(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);
        return EXIT_FAILURE;
    }
    MPI_Finalize();
    return status;
}
