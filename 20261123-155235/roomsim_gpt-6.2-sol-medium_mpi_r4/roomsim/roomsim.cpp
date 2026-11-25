/**
 * Room Response Simulation Benchmark
 * 
 * This is an MPI implementation of simplified room impulse response
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
#include <cinttypes>
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
    bool rayIntersectsBox(const Vec3& midpoint, const Vec3& d, const Vec3& ad) const {
        const Vec3 c = midpoint - center;

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
    bool applyToTris(const Vec3& midpoint, const Vec3& d, const Vec3& ad, const Func& func) const {
        // If leaf node, check triangles directly
        if (!triangleIndices.empty()) {
            for (size_t idx : triangleIndices) {
                if (func(idx, (*allTriangles)[idx])) return true;
            }
            return false;
        }

        // Otherwise, descend to children
        for (int i = 0; i < 8; ++i) {
            if (children[i] && children[i]->rayIntersectsBox(midpoint, d, ad)) {
                if (children[i]->applyToTris(midpoint, d, ad, func)) return true;
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

// MT19937 with access to its raw state so rank offsets can be reached without
// generating all random numbers assigned to preceding ranks.
class Mt19937 {
public:
    using result_type = uint32_t;

    explicit Mt19937(result_type seed) {
        state[0] = seed;
        for (size_t i = 1; i < state.size(); ++i) {
            const result_type prev = state[i - 1];
            state[i] = std::mt19937::initialization_multiplier * (prev ^ (prev >> 30)) + i;
        }
    }

    static constexpr result_type min() { return std::mt19937::min(); }
    static constexpr result_type max() { return std::mt19937::max(); }

    result_type operator()() {
        result_type x = nextRaw();
        x ^= x >> 11;
        x ^= (x << 7) & 0x9d2c5680u;
        x ^= (x << 15) & 0xefc60000u;
        return x ^ (x >> 18);
    }

    void discard(uint64_t draws) {
        // Twisting whole blocks is cheaper than polynomial arithmetic for
        // short skips. Bound this work independently of the problem size.
        if (draws < (uint64_t{1} << 27)) {
            while (draws > state.size() - cursor) {
                draws -= state.size() - cursor;
                twist();
            }
            cursor += draws;
            return;
        }

        // Reduce x^draws modulo the characteristic polynomial of the
        // one-word MT19937 transition, by repeated squaring over GF(2).
        // The leading x^19937 coefficient is implicit.
        constexpr int degree = 19937;
        constexpr std::array<int, 134> lowerPowers{
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
            19087, 19314
        };
        std::bitset<2 * degree> polynomial;
        polynomial[0] = true;
        for (int bit = std::bit_width(draws); bit-- > 0;) {
            std::bitset<2 * degree> next;
            const int digit = (draws >> bit) & 1;
            for (int i = 0; i < degree; ++i) next[2 * i + digit] = polynomial[i];
            for (int i = 2 * degree - 1; i >= degree; --i) {
                if (!next[i]) continue;
                next[i] = false;
                for (int power : lowerPowers) next[i - degree + power].flip();
            }
            polynomial = next;
        }

        // Apply the polynomial to consecutive raw outputs to reconstruct the
        // 624 words starting at the target position, then resume normal draws.
        std::array<result_type, degree + std::mt19937::state_size - 1> prefix;
        for (auto& word : prefix) word = nextRaw();
        state.fill(0);
        for (int i = 0; i < degree; ++i) {
            if (!polynomial[i]) continue;
            for (size_t j = 0; j < state.size(); ++j) state[j] ^= prefix[i + j];
        }
        cursor = 0;
    }

private:
    std::array<result_type, std::mt19937::state_size> state;
    size_t cursor = state.size();

    void twist() {
        constexpr size_t n = std::mt19937::state_size;
        constexpr size_t m = std::mt19937::shift_size;
        auto mix = [](result_type a, result_type b) {
            const result_type joined = (a & 0x80000000u) | (b & 0x7fffffffu);
            return (joined >> 1) ^ ((joined & 1) ? std::mt19937::xor_mask : 0);
        };
        for (size_t i = 0; i < n - m; ++i) state[i] = state[i + m] ^ mix(state[i], state[i + 1]);
        for (size_t i = n - m; i < n - 1; ++i) state[i] = state[i + m - n] ^ mix(state[i], state[i + 1]);
        state[n - 1] = state[m - 1] ^ mix(state[n - 1], state[0]);
        cursor = 0;
    }

    result_type nextRaw() {
        if (cursor == state.size()) twist();
        return state[cursor++];
    }
};

class RandomGenerator {
    Mt19937 rng;
    std::uniform_real_distribution<val_t> dist;
public:
    explicit RandomGenerator(uint32_t seed = 42) : rng(seed), dist(0.0f, 1.0f) {}
    val_t rand() { return dist(rng); }
    void discard(uint64_t draws) { rng.discard(draws); }
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
    const Vec3 half = dir * 0.5f;
    const Vec3 absHalf = {std::abs(half.x), std::abs(half.y), std::abs(half.z)};

    return octree.applyToTris(from + half, half, absHalf, [&](size_t idx, const Triangle& tri) {
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

// Cull triangles facing the same direction before drawing ray samples.
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
// Tau (time delay) Computation
// ============================================================================

int computeTau(const Vec3& centerI, const Vec3& centerJ) {
    val_t dist = (centerI - centerJ).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct RowPartition {
    MPI_Comm comm;
    int rank;
    std::vector<int> counts;
    std::vector<int> offsets;
    int firstRow;
    int localRows;

    RowPartition(int numTriangles, MPI_Comm communicator) : comm(communicator) {
        int ranks;
        MPI_Comm_rank(comm, &rank);
        MPI_Comm_size(comm, &ranks);
        counts.resize(ranks);
        offsets.resize(ranks);
        int offset = 0;
        for (int r = 0; r < ranks; ++r) {
            counts[r] = numTriangles / ranks + (r < numTriangles % ranks);
            offsets[r] = offset;
            offset += counts[r];
        }
        firstRow = offsets[rank];
        localRows = counts[rank];
    }
};

struct Interaction {
    idx_t lookback;  // delay * numTriangles - emitter index
    val_t weight;
};

struct SimulationState {
    std::vector<Triangle> triangles;
    size_t numTimesteps;
    size_t sourceIndex;
    val_t reflectivity;

    // Each rank owns a contiguous block of receiver rows.
    std::vector<std::vector<Interaction>> interactions;
    std::vector<val_t> localRadiosity;
    std::vector<val_t> sourceRadiosity;
    std::vector<val_t> distances;
    uint64_t nonZeroKij = 0;
    int maxDelay = 0;

    SimulationState(int subdivisions, size_t timesteps, size_t sourceIdx, val_t rho)
        : triangles(IcosphereMesh(subdivisions, ROOM_RADIUS).triangles),
          numTimesteps(timesteps), sourceIndex(sourceIdx % triangles.size()),
          reflectivity(rho) {}
};

// ============================================================================
// Distributed Precomputation
// ============================================================================

void precomputeInteractions(SimulationState& state, const RowPartition& rows, const Octree& octree) {
    const size_t n = state.triangles.size();
    const bool root = rows.rank == 0;
    if (root) {
        printf("Computing time delays (Tau)...\n");
        printf("Computing form factors (Kij)...\n");
    }
    std::vector<val_t> areas(n);
    std::vector<Vec3> centers(n);
    for (size_t j = 0; j < n; ++j) {
        areas[j] = state.triangles[j].area();
        centers[j] = state.triangles[j].center();
    }

    // Every sampled pair uses four draws per ray. Prefix counts locate each
    // rank in the original row-major random stream after skipping culled pairs.
    uint64_t precedingPairs = 0;
    if (rows.counts.size() > 1) {
        uint64_t localPairs = 0;
        for (int local = 0; local < rows.localRows; ++local) {
            const size_t i = rows.firstRow + local;
            for (size_t j = 0; j < n; ++j) {
                localPairs += i != j && samplesPair(state.triangles[i], state.triangles[j]);
            }
        }
        MPI_Exscan(&localPairs, &precedingPairs, 1, MPI_UINT64_T, MPI_SUM, rows.comm);
        if (root) precedingPairs = 0;
    }
    RandomGenerator rng(42);
    rng.discard(precedingPairs * (4 * NUM_RAYS));

    state.interactions.resize(rows.localRows);
    int localMaxDelay = 0;
    // With enough timesteps to span the room, almost all pairs interact.
    const auto roomDelay = static_cast<size_t>(2 * ROOM_RADIUS * INV_WAVE_SPEED);
    for (int local = 0; local < rows.localRows; ++local) {
        const size_t i = rows.firstRow + local;
        auto& interactions = state.interactions[local];
        if (state.numTimesteps > roomDelay) interactions.reserve(n - 1);
        for (size_t j = 0; j < n; ++j) {
            if (i == j || !samplesPair(state.triangles[i], state.triangles[j])) continue;

            const val_t kij = computeKij(i, j, state.triangles, octree, rng);
            state.nonZeroKij += kij > EPSILON;
            if (kij <= ZERO) continue;

            // Distinct triangles in the generated mesh have distinct centers.
            const int delay = computeTau(centers[i], centers[j]);
            if (static_cast<size_t>(delay) >= state.numTimesteps) continue;

            localMaxDelay = std::max(localMaxDelay, delay);
            interactions.push_back({
                static_cast<idx_t>(delay * n - j), std::min(kij * areas[j], ONE)
            });
        }
    }
    MPI_Allreduce(&localMaxDelay, &state.maxDelay, 1, MPI_INT, MPI_MAX, rows.comm);
    if (root) printf("  Progress: %zu/%zu triangles\n", n, n);
}

// ============================================================================
// Distributed Wave Propagation
// ============================================================================

void runSimulation(SimulationState& state, const RowPartition& rows) {
    const bool root = rows.rank == 0;
    if (root) printf("Running wave propagation simulation...\n");
    const size_t n = state.triangles.size();
    const size_t timesteps = state.numTimesteps;
    state.localRadiosity.resize(timesteps * rows.localRows);
    state.sourceRadiosity.resize(timesteps);

    // Only retain the global timesteps that can still contribute. The mirror
    // makes the lookbacks contiguous across the wrap point. The untouched
    // rows start at zero, so lookbacks before timestep zero contribute nothing.
    const size_t historyLength = state.maxDelay + 1;
    const size_t historySize = historyLength * n;
    std::vector<val_t> history(2 * historySize, ZERO);
    const bool equalBlocks = n % rows.counts.size() == 0;
    size_t slot = 0;

    for (size_t t = 0; t < timesteps; ++t) {
        val_t* gathered = history.data() + slot * n;
        val_t* current = gathered + historySize;
        val_t* local = state.localRadiosity.data() + t * rows.localRows;

        for (int i = 0; i < rows.localRows; ++i) {
            val_t sumB = ZERO;
            for (const auto& interaction : state.interactions[i]) {
                const val_t radJ = *(current - interaction.lookback);
                if (radJ <= ZERO) continue;
                // GCC rounds the product separately in the original
                // accumulation with its inline weight clamp.
#if __has_builtin(__builtin_assoc_barrier)
                sumB += __builtin_assoc_barrier(interaction.weight * radJ);
#else
                sumB += std::min(interaction.weight, ONE) * radJ;
#endif
            }

            const val_t emission = t < timesteps / 2 &&
                static_cast<size_t>(rows.firstRow + i) == state.sourceIndex ? ONE : ZERO;
            local[i] = state.reflectivity * sumB + emission;
        }

        if (equalBlocks) {
            MPI_Allgather(local, rows.localRows, MPI_FLOAT, gathered, rows.localRows,
                          MPI_FLOAT, rows.comm);
        } else {
            MPI_Allgatherv(local, rows.localRows, MPI_FLOAT, gathered,
                           rows.counts.data(), rows.offsets.data(), MPI_FLOAT, rows.comm);
        }
        state.sourceRadiosity[t] = gathered[state.sourceIndex];
        std::copy_n(gathered, n, current);
        if (++slot == historyLength) slot = 0;

        if (root && ((t + 1) % 10 == 0 || t + 1 == timesteps)) {
            printf("  Timestep %zu/%zu\n", t + 1, timesteps);
        }
    }
}

// ============================================================================
// Distributed Distance Computation
// ============================================================================

void computeDistances(SimulationState& state, const RowPartition& rows) {
    if (rows.rank == 0) printf("Computing distances via cross-correlation...\n");

    const size_t timesteps = state.numTimesteps;
    std::vector<val_t> localDistances(rows.localRows);
    std::vector<val_t> trace(timesteps);
    for (int i = 0; i < rows.localRows; ++i) {
        for (size_t t = 0; t < timesteps; ++t) {
            trace[t] = state.localRadiosity[t * rows.localRows + i];
        }

        val_t maxCorr = ZERO;
        size_t bestT = 0;
        for (size_t t = 0; t < timesteps; ++t) {
            val_t sum = ZERO;
            for (size_t tt = 0; tt < timesteps - t; ++tt) {
                sum += state.sourceRadiosity[tt] * trace[tt + t];
            }
            if (sum > maxCorr) {
                maxCorr = sum;
                bestT = t;
            }
        }
        localDistances[i] = WAVE_SPEED * static_cast<val_t>(bestT);
    }

    if (rows.rank == 0) state.distances.resize(state.triangles.size());
    MPI_Gatherv(localDistances.data(), rows.localRows, MPI_FLOAT, state.distances.data(),
                rows.counts.data(), rows.offsets.data(), MPI_FLOAT, 0, rows.comm);
}

// ============================================================================
// Validation
// ============================================================================

std::array<uint64_t, 2> collectValidationCounts(const SimulationState& state,
                                                const RowPartition& rows) {
    uint64_t receivedEnergy = 0;
    for (int i = 0; i < rows.localRows; ++i) {
        for (size_t t = 0; t < state.numTimesteps; ++t) {
            if (state.localRadiosity[t * rows.localRows + i] > EPSILON) {
                ++receivedEnergy;
                break;
            }
        }
    }
    const std::array<uint64_t, 2> localCounts{receivedEnergy, state.nonZeroKij};
    std::array<uint64_t, 2> counts{};
    MPI_Reduce(localCounts.data(), counts.data(), static_cast<int>(counts.size()),
               MPI_UINT64_T, MPI_SUM, 0, rows.comm);
    return counts;
}

bool validateResults(const SimulationState& state, uint64_t receivedEnergy, uint64_t nonZeroKij) {
    const size_t n = state.triangles.size();
    printf("\nValidation:\n");

    // Check that distances are non-negative
    bool allNonNegative = true;
    val_t minDist = std::numeric_limits<val_t>::max();
    val_t maxDist = std::numeric_limits<val_t>::lowest();
    val_t sumDist = ZERO;
    int nonZeroCount = 0;

    for (size_t i = 0; i < state.triangles.size(); ++i) {
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
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.triangles.size()));
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, state.triangles.size());

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    printf("  Triangles receiving energy: %" PRIu64 "/%zu\n", receivedEnergy, n);
    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    printf("  Non-zero form factors: %" PRIu64 "/%zu (%.2f%%)\n",
           nonZeroKij, n * n, 100.0f * nonZeroKij / static_cast<val_t>(n * n));

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
    for (size_t i = 0; i < state.triangles.size(); ++i) {
        const auto bits = std::bit_cast<uint32_t>(state.distances[i]);
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

int runBenchmark(int subdivisions, int timesteps, int sourceIdx, val_t reflectivity,
                 bool validate, bool printResults, MPI_Comm comm) {
    SimulationState state(subdivisions, timesteps, static_cast<size_t>(sourceIdx), reflectivity);
    const size_t n = state.triangles.size();
    const size_t t = state.numTimesteps;
    const RowPartition rows(static_cast<int>(n), comm);
    const bool root = rows.rank == 0;

    if (root) {
        printf("Generated icosphere mesh with %zu triangles\n", n);
        printf("Building octree...\n\n");
    }
    long preDuration;
    {
        Octree octree;
        octree.build(state.triangles);
        MPI_Barrier(comm);
        const double startPre = MPI_Wtime();
        precomputeInteractions(state, rows, octree);
        preDuration = static_cast<long>((MPI_Wtime() - startPre) * 1000.0);
    }
    if (root) printf("Precomputation time: %ld ms\n\n", preDuration);

    const double startSim = MPI_Wtime();
    runSimulation(state, rows);
    const double endSim = MPI_Wtime();
    const auto simDuration = static_cast<long>((endSim - startSim) * 1000.0);
    if (root) printf("Simulation time: %ld ms\n\n", simDuration);

    const double startDist = MPI_Wtime();
    computeDistances(state, rows);
    const double endDist = MPI_Wtime();
    const auto distDuration = static_cast<long>((endDist - startDist) * 1000.0);

    uint64_t localMemory = state.interactions.capacity() * sizeof(std::vector<Interaction>);
    for (const auto& row : state.interactions) {
        localMemory += row.capacity() * sizeof(Interaction);
    }
    localMemory += (state.localRadiosity.capacity() + state.sourceRadiosity.capacity() +
                    2 * (state.maxDelay + 1) * n) * sizeof(val_t);
    uint64_t maxMemory = 0;
    MPI_Reduce(&localMemory, &maxMemory, 1, MPI_UINT64_T, MPI_MAX, 0, comm);

    std::array<uint64_t, 2> validationCounts{};
    if (validate) validationCounts = collectValidationCounts(state, rows);
    const long localTotalTime = preDuration + simDuration + distDuration;
    long totalTime = 0;
    MPI_Reduce(&localTotalTime, &totalTime, 1, MPI_LONG, MPI_MAX, 0, comm);
    if (!root) return 0;
    printf("Distance computation time: %ld ms\n\n", distDuration);
    printf("Total computation time: %ld ms\n", totalTime);

    const double kijOps = static_cast<double>(n) * n;
    const double simOps = kijOps * t;
    const double distOps = static_cast<double>(n) * t * t;
    printf("\nPerformance:\n");
    printf("  Triangles: %zu\n", n);
    printf("  Timesteps: %zu\n", t);
    printf("  Form factor computations: %.2e\n", kijOps);
    printf("  Simulation operations: %.2e\n", simOps);
    printf("  Distance computations: %.2e\n", distOps);
    printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);
    printf("  Memory usage (largest rank): %.2f MB\n", maxMemory / (1024.0 * 1024.0));
    printf("  Result hash: %016" PRIX64 "\n\n", computeHash(state));

    if (printResults) {
        std::vector<double> distData(state.distances.begin(), state.distances.end());
        print_results(distData, "Distances");
    }

    return validate && !validateResults(state, validationCounts[0], validationCounts[1]);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    const bool root = rank == 0;
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
            if (root) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (root) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }
    if (timesteps < 0) {
        if (root) fprintf(stderr, "Number of timesteps must be non-negative\n");
        MPI_Finalize();
        return 1;
    }

    const int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);
    if (root) {
        printf("Room Response Simulation Benchmark\n");
        printf("===================================\n");
        printf("Target triangles: %d (using %d subdivisions)\n", targetTriangles, subdivisions);
        printf("Timesteps: %d\n", timesteps);
        printf("Source triangle: %d\n", sourceIdx);
        printf("Reflectivity: %.2f\n", reflectivity);
        printf("Validation: %s\n\n", validate ? "enabled" : "disabled");
    }

    const int numTriangles = 20 << (2 * subdivisions);
    MPI_Comm active = MPI_COMM_WORLD;
    if (ranks > numTriangles) {
        MPI_Comm_split(MPI_COMM_WORLD, rank < numTriangles ? 0 : MPI_UNDEFINED, rank, &active);
    }
    int result = 0;
    if (active != MPI_COMM_NULL) {
        result = runBenchmark(subdivisions, timesteps, sourceIdx, reflectivity,
                              validate, printResults, active);
        if (active != MPI_COMM_WORLD) MPI_Comm_free(&active);
    }
    MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return result;
}
