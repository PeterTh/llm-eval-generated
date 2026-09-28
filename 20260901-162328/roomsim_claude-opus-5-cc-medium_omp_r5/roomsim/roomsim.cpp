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
#include <new>
#include <random>
#include <type_traits>
#include <utility>
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
// Storage for the large matrices
// ============================================================================

// Allocator that skips value initialization, so that the big matrices can be
// first touched (and therefore NUMA placed) in parallel by the threads that
// later work on them.
template <typename T>
struct NoInitAllocator {
    using value_type = T;

    NoInitAllocator() = default;
    template <typename U>
    constexpr NoInitAllocator(const NoInitAllocator<U>&) noexcept {}

    T* allocate(size_t n) { return static_cast<T*>(::operator new(n * sizeof(T))); }
    void deallocate(T* p, size_t) noexcept { ::operator delete(static_cast<void*>(p)); }

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        if constexpr (sizeof...(Args) != 0 || !std::is_trivially_default_constructible_v<U>) {
            ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
    }

    template <typename U>
    bool operator==(const NoInitAllocator<U>&) const noexcept { return true; }
    template <typename U>
    bool operator!=(const NoInitAllocator<U>&) const noexcept { return false; }
};

template <typename T>
using Matrix = std::vector<T, NoInitAllocator<T>>;

// Allocate and zero a matrix. The zeroing is done with the same static block
// distribution that the compute phases use, so every page ends up on the NUMA
// node of the thread that will work on it.
template <typename T>
void allocateZeroed(Matrix<T>& m, size_t n) {
    m.resize(n);
    T* __restrict data = m.data();
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; ++i) {
        data[i] = T{};
    }
}

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

    // Flattened, cache friendly copy of the tree used for the (extremely hot)
    // visibility queries. Children of a node are stored contiguously and leaf
    // triangles are inlined, so a query never chases pointers.
    struct FlatNode {
        Vec3 center;
        Vec3 halfExtent;
        uint32_t firstChild;
        uint32_t childCount;
        uint32_t triStart;
        uint32_t triCount;
    };
    struct FlatTri {
        Vec3 v0;        // first vertex
        Vec3 e1, e2;    // edge vectors, precomputed for the intersection test
        uint32_t idx;
    };
    std::vector<FlatNode> flatNodes;
    std::vector<FlatTri> flatTris;

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
        flatten();
    }

private:
    // Breadth first linearization of the tree
    void flatten() {
        flatNodes.clear();
        flatTris.clear();

        std::vector<const Octree*> order;
        order.push_back(this);
        flatNodes.push_back(FlatNode{center, halfExtent, 0, 0, 0, 0});

        for (size_t head = 0; head < order.size(); ++head) {
            const Octree* node = order[head];

            const uint32_t triStart = static_cast<uint32_t>(flatTris.size());
            for (size_t idx : node->triangleIndices) {
                const Triangle& tri = (*allTriangles)[idx];
                flatTris.push_back(FlatTri{tri.a, tri.b - tri.a, tri.c - tri.a,
                                           static_cast<uint32_t>(idx)});
            }
            flatNodes[head].triStart = triStart;
            flatNodes[head].triCount = static_cast<uint32_t>(node->triangleIndices.size());

            const uint32_t firstChild = static_cast<uint32_t>(flatNodes.size());
            uint32_t childCount = 0;
            for (int i = 0; i < 8; ++i) {
                if (node->children[i]) {
                    order.push_back(node->children[i].get());
                    flatNodes.push_back(FlatNode{node->children[i]->center,
                                                node->children[i]->halfExtent, 0, 0, 0, 0});
                    ++childCount;
                }
            }
            flatNodes[head].firstChild = firstChild;
            flatNodes[head].childCount = childCount;
        }
    }

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
    // Check if a ray intersects the given node's bounding box
    static bool rayIntersectsBox(const FlatNode& node, const Vec3& mid, const Vec3& d, const Vec3& ad) {
        const Vec3& center = node.center;
        const Vec3& halfExtent = node.halfExtent;
        Vec3 c = mid - center;

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
        // Ray extent, shared by all box tests of this query
        const Vec3 d = (p2 - p1) * 0.5f;
        const Vec3 mid = p1 + d;
        const Vec3 ad = {std::abs(d.x), std::abs(d.y), std::abs(d.z)};

        // The tree is at most ~8 levels deep (nodes below MAX_OCTREE_LEAF_SIZE
        // become leaves), so at most 8 children per level fit comfortably.
        uint32_t stack[128];
        int sp = 0;
        stack[sp++] = 0;  // root, never culled (matches the original traversal)

        while (sp > 0) {
            const FlatNode& node = flatNodes[stack[--sp]];

            // If leaf node, check triangles directly
            if (node.triCount != 0) {
                const FlatTri* tris = flatTris.data() + node.triStart;
                for (uint32_t k = 0; k < node.triCount; ++k) {
                    if (func(tris[k])) return true;
                }
                continue;
            }

            // Otherwise, descend to children
            for (uint32_t c = node.firstChild; c < node.firstChild + node.childCount; ++c) {
                if (rayIntersectsBox(flatNodes[c], mid, d, ad)) {
                    stack[sp++] = c;
                }
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
    // Resume the sequential stream from a previously captured engine state
    explicit RandomGenerator(const std::mt19937& state) : rng(state), dist(0.0f, 1.0f) {}
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

// Takes the triangle as origin vertex plus its two edge vectors (e1 = v1 - v0,
// e2 = v2 - v0). The rejection tests are evaluated branch-free: the arithmetic is
// identical to the early-out version, only the decision is deferred to the end.
inline val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                                  const Vec3& v0, const Vec3& e1, const Vec3& e2) {
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;

    val_t t = e2.dot(qvec) * invDet;

    const bool hit = std::abs(det) >= EPSILON && u >= 0.0f && u <= 1.0f &&
                     v >= 0.0f && u + v <= 1.0f;
    return hit ? t : std::numeric_limits<val_t>::max();
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

    return octree.applyToTris(from, to, [&](const Octree::FlatTri& tri) {
        if (tri.idx == srcTriIdx || tri.idx == dstTriIdx) return false;

        val_t dist = rayTriangleIntersect(from, dirNorm, tri.v0, tri.e1, tri.e2);
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

// Triangles facing the same direction are culled before any random number is drawn
bool isKijCulled(const Triangle& triI, const Triangle& triJ) {
    return triI.normal().dot(triJ.normal()) > 0.99f;
}

// Number of random numbers a non-culled computeKij() call draws from the stream:
// two points per ray, two uniform samples per point.
constexpr uint64_t KIJ_RANDOM_DRAWS = 4 * static_cast<uint64_t>(NUM_RAYS);

// Compute form factor Kij between triangle i (receiver) and triangle j (emitter)
val_t computeKij(size_t idxI, size_t idxJ,
                  const std::vector<Triangle>& triangles,
                  const Octree& octree,
                  RandomGenerator& rng) {
    const Triangle& triI = triangles[idxI];
    const Triangle& triJ = triangles[idxJ];

    // Cull triangles facing the same direction
    if (isKijCulled(triI, triJ)) return ZERO;

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

inline int computeTau(const Vec3& centerI, const Vec3& centerJ) {
    val_t dist = (centerI - centerJ).norm();
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
    Matrix<val_t> kij;              // Form factors (N x N matrix, row-major)
    Matrix<int> tau;                // Time delays (N x N matrix, row-major)
    Matrix<val_t> radE;             // Emission radiosity (T x N matrix)
    Matrix<val_t> radB;             // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure

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
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    // Initialize matrices (allocated and first touched in parallel)
    allocateZeroed(state.kij, state.numTriangles * state.numTriangles);
    allocateZeroed(state.tau, state.numTriangles * state.numTriangles);
    allocateZeroed(state.radE, timesteps * state.numTriangles);
    allocateZeroed(state.radB, timesteps * state.numTriangles);
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

// The form factor computation consumes a single sequential random stream, which makes
// the (i, j) pairs data dependent on each other. To parallelize it while reproducing the
// original stream bit-exactly, the pair space is split into fixed size chunks and the
// engine state at the start of each chunk is captured. The number of draws a pair
// consumes only depends on the (cheap) culling test, so the per-chunk draw counts can be
// computed up front in parallel; a single thread then walks the engine forward (using the
// cheap state advance of discard(), without generating any variates) and publishes the
// chunk start states while all other threads already consume them.
void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");

    const size_t N = state.numTriangles;
    const size_t numPairs = N * N;
    if (numPairs == 0) return;

    const std::vector<Triangle>& triangles = state.triangles;

    // Chunk the flattened pair space finely enough for dynamic load balancing.
    const size_t targetChunks = static_cast<size_t>(omp_get_max_threads()) * 32;
    const size_t chunkSize = std::max<size_t>(64, (numPairs + targetChunks - 1) / targetChunks);
    const size_t numChunks = (numPairs + chunkSize - 1) / chunkSize;

    // Random draws consumed by each chunk
    std::vector<uint64_t> chunkDraws(numChunks);
#pragma omp parallel for schedule(static)
    for (size_t c = 0; c < numChunks; ++c) {
        const size_t begin = c * chunkSize;
        const size_t end = std::min(begin + chunkSize, numPairs);
        uint64_t draws = 0;
        for (size_t p = begin; p < end; ++p) {
            const size_t i = p / N;
            const size_t j = p - i * N;
            if (i == j) continue;
            if (!isKijCulled(triangles[i], triangles[j])) draws += KIJ_RANDOM_DRAWS;
        }
        chunkDraws[c] = draws;
    }

    std::vector<std::mt19937> chunkStates(numChunks);
    std::atomic<size_t> statesReady{0};
    std::atomic<size_t> nextChunk{0};
    val_t* __restrict kijData = state.kij.data();

#pragma omp parallel
    {
        // One thread produces the chunk start states, then joins the workers.
        if (omp_get_thread_num() == 0) {
            std::mt19937 gen(42);
            for (size_t c = 0; c < numChunks; ++c) {
                chunkStates[c] = gen;
                statesReady.store(c + 1, std::memory_order_release);
                gen.discard(chunkDraws[c]);
            }
        }

        for (;;) {
            const size_t c = nextChunk.fetch_add(1, std::memory_order_relaxed);
            if (c >= numChunks) break;

            // Wait for the producer to publish this chunk's engine state. It walks
            // the stream several times faster than the workers consume it, so this
            // only spins during the very first chunks.
            while (statesReady.load(std::memory_order_acquire) <= c) {
#if defined(__x86_64__) || defined(__i386__)
                __builtin_ia32_pause();
#endif
            }

            RandomGenerator rng(chunkStates[c]);
            const size_t begin = c * chunkSize;
            const size_t end = std::min(begin + chunkSize, numPairs);
            for (size_t p = begin; p < end; ++p) {
                const size_t i = p / N;
                const size_t j = p - i * N;
                if (i == j) continue;
                kijData[p] = computeKij(i, j, triangles, state.octree, rng);
            }
        }
    }

    for (size_t i = 0; i < N; ++i) {
        if ((i + 1) % 100 == 0 || i + 1 == N) {
            printf("  Progress: %zu/%zu triangles\n", i + 1, N);
        }
    }
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");

    const size_t N = state.numTriangles;

    // Triangle centers are reused for every pair
    std::vector<Vec3> centers(N);
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        centers[i] = state.triangles[i].center();
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        int* tauRow = &state.tau[state.idx2d(i, 0)];
        const Vec3 ci = centers[i];
        for (size_t j = 0; j < N; ++j) {
            if (i == j) continue;
            tauRow[j] = computeTau(ci, centers[j]);
        }
    }
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");

    const size_t N = state.numTriangles;
    const val_t* __restrict areas = state.areas.data();
    const val_t* __restrict rho = state.rho.data();
    const val_t* __restrict kijMat = state.kij.data();
    const int* __restrict tauMat = state.tau.data();
    const val_t* __restrict radE = state.radE.data();
    val_t* __restrict radB = state.radB.data();

    // Time steps are sequentially dependent (a triangle can only receive radiosity
    // emitted at strictly earlier steps), the triangles of one step are independent.
#pragma omp parallel
    for (size_t t = 0; t < state.numTimesteps; ++t) {
#pragma omp for schedule(static)
        for (size_t i = 0; i < N; ++i) {
            val_t sumB = ZERO;

            const val_t* __restrict kijRow = kijMat + i * N;
            const int* __restrict tauRow = tauMat + i * N;

            for (size_t j = 0; j < N; ++j) {
                if (i == j) continue;

                int tauij = tauRow[j];

                // Skip if wave hasn't yet propagated from j to i
                if (static_cast<int>(t) < tauij) continue;

                val_t kij = kijRow[j];
                if (kij <= ZERO) continue;

                // Get radiosity from source triangle at time when emission occurred
                size_t srcTime = t - static_cast<size_t>(tauij);
                val_t radJ = radB[srcTime * N + j];
                if (radJ <= ZERO) continue;

                // Accumulate contribution: form factor * area * source radiosity
                sumB += std::min(kij * areas[j], ONE) * radJ;
            }

            // Update radiosity: reflection + emission
            radB[t * N + i] = rho[i] * sumB + radE[t * N + i];
        }
        // implicit barrier of the worksharing loop separates the time steps

        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
#pragma omp master
            printf("  Timestep %zu/%zu\n", t + 1, state.numTimesteps);
        }
    }
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");

    const size_t N = state.numTriangles;
    const size_t T = state.numTimesteps;

    // Source signal is the same for every triangle
    std::vector<val_t> srcSignal(T);
#pragma omp parallel for schedule(static)
    for (size_t t = 0; t < T; ++t) {
        srcSignal[t] = state.radB[state.idxTN(t, state.sourceIndex)];
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < N; ++i) {
        val_t maxCorr = ZERO;
        int bestT = 0;

        // Discrete cross-correlation to find time delay
        for (size_t t = 0; t < T; ++t) {
            val_t sum = ZERO;

            for (size_t tt = t; tt < T; ++tt) {
                val_t pB = state.radB[tt * N + i];
                val_t pS = srcSignal[tt - t];
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
#pragma omp parallel for schedule(static) reduction(+ : receivedEnergy)
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
#pragma omp parallel for schedule(static) reduction(+ : nonZeroKij)
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
