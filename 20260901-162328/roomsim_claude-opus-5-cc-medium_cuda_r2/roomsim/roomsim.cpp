/**
 * Room Response Simulation Benchmark -- CUDA (GPU) parallel implementation
 *
 * This program simulates room impulse responses using radiosity-based wave
 * propagation. It models how sound/light waves propagate between surfaces in a
 * room by computing:
 *
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 *
 * The implementation generates a procedural icosphere mesh to simulate a room.
 *
 * ----------------------------------------------------------------------------
 * Parallelization notes (CUDA)
 * ----------------------------------------------------------------------------
 * All three compute phases run on the GPU:
 *
 *  * Tau (N x N)         : one thread per matrix entry.
 *  * Kij (N x N)         : one CUDA block per (receiver triangle, tile of
 *                          emitters), one thread per emitter; every thread
 *                          traces the 16 visibility rays of its pair
 *                          sequentially (so the per-pair accumulation order is
 *                          unchanged).  The octree is flattened into arrays and
 *                          traversed with an explicit stack.  Because
 *                          `isRayBlocked` only asks "does *any* triangle occlude
 *                          the segment", the order in which candidate triangles
 *                          are visited does not influence the result.
 *  * Wave propagation    : timesteps are inherently sequential (tau >= 1 for
 *                          i != j, so every timestep only reads strictly older
 *                          radiosities), but all N receivers of a timestep are
 *                          independent -> one thread per receiver, each summing
 *                          over j in the original order.  Kij/Tau are stored
 *                          transposed and interleaved so that this access
 *                          pattern is coalesced and needs one load per emitter.
 *  * Cross-correlation   : one block per triangle, one thread per lag t, each
 *                          accumulating over tt in the original order.
 *
 * Every phase keeps the reference implementation's floating point operation
 * order (and matches the host compiler's fma contraction, see Vec3::dot), so
 * the results are bit-identical to the sequential version.
 *
 * Bit-exact random number equivalence: the reference implementation draws from
 * a single sequential std::mt19937 stream, consuming exactly 64 words per
 * non-culled triangle pair (2 barycentric coordinates for each of the two
 * sample points, for each of the 16 rays) and none for culled pairs.  The
 * per-row number of non-culled pairs is therefore computed first, which yields
 * every pair's absolute offset into the MT19937 stream.  The stream itself is
 * generated in bulk on the host (it is strictly sequential by construction) and
 * streamed to the device in double-buffered chunks that overlap with kernel
 * execution; the device reproduces libstdc++'s
 * uniform_real_distribution<float>(0,1) word-for-word.
 */

#include <algorithm>
#include <array>
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

#include <cuda_runtime.h>

#include "../common/results_output.hpp"

// ============================================================================
// CUDA helpers
// ============================================================================

#define CUDA_CHECK(call)                                                                   \
    do {                                                                                   \
        cudaError_t err_ = (call);                                                         \
        if (err_ != cudaSuccess) {                                                         \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,               \
                    cudaGetErrorString(err_));                                             \
            exit(EXIT_FAILURE);                                                            \
        }                                                                                  \
    } while (0)

#define HD __host__ __device__

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

// Random words consumed by one non-culled triangle pair (4 per ray).
constexpr int WORDS_PER_PAIR = 4 * NUM_RAYS;

// Maximum size (in 32-bit words) of one host->device random number chunk.
constexpr size_t RNG_CHUNK_WORDS = 16u << 20;  // 64 MiB per staging buffer

// Thread block sizes.
constexpr int KIJ_BLOCK = 128;
constexpr int SIM_BLOCK = 64;
constexpr int DIST_BLOCK = 256;
constexpr int RED_BLOCK = 256;

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

    // The device variants below are algebraically the same expressions, but
    // they pin down the floating point contraction so that the GPU reproduces
    // the host compiler's choices bit for bit: in `p1 (+|-) p2` with two
    // products, gcc rounds p2 and fuses p1 into the add.
    HD val_t dot(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return __fmaf_rn(z, o.z, __fmaf_rn(x, o.x, __fmul_rn(y, o.y)));
#else
        return x * o.x + y * o.y + z * o.z;
#endif
    }
    HD Vec3 cross(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return {__fmaf_rn(y, o.z, -__fmul_rn(z, o.y)),
                __fmaf_rn(z, o.x, -__fmul_rn(x, o.z)),
                __fmaf_rn(x, o.y, -__fmul_rn(y, o.x))};
#else
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
#endif
    }

    HD val_t squaredNorm() const {
#ifdef __CUDA_ARCH__
        return __fmaf_rn(z, z, __fmaf_rn(x, x, __fmul_rn(y, y)));
#else
        return x * x + y * y + z * z;
#endif
    }
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
};

// ============================================================================
// Flattened (GPU resident) octree
// ============================================================================

// Node layout: 80 bytes, 16-byte aligned.  A node is a leaf iff triCount > 0
// (mirroring `!triangleIndices.empty()` in the reference traversal).
struct __align__(16) GNode {
    float4 c;                    // box center (xyz)
    float4 h;                    // box half extent (xyz)
    int child[8];                // child node index or -1
    int triStart, triCount;      // range in the flat triangle index list
    int pad0, pad1;
};

// Form factor and time delay of one triangle pair, stored together so that the
// wave propagation kernel needs a single 8-byte load per emitter.
struct __align__(8) KijTau {
    val_t kij;
    int tau;
};

// Depth-first flattening; returns the index of the emitted node.
static int flattenOctree(const Octree* node, std::vector<GNode>& nodes,
                         std::vector<int>& triList, int depth, int& maxDepth) {
    const int id = static_cast<int>(nodes.size());
    nodes.emplace_back();
    if (depth > maxDepth) maxDepth = depth;

    GNode n;
    memset(&n, 0, sizeof(n));
    n.c = make_float4(node->center.x, node->center.y, node->center.z, 0.0f);
    n.h = make_float4(node->halfExtent.x, node->halfExtent.y, node->halfExtent.z, 0.0f);
    for (int k = 0; k < 8; ++k) n.child[k] = -1;

    if (!node->triangleIndices.empty()) {
        n.triStart = static_cast<int>(triList.size());
        n.triCount = static_cast<int>(node->triangleIndices.size());
        for (size_t idx : node->triangleIndices) triList.push_back(static_cast<int>(idx));
        nodes[id] = n;
    } else {
        // Children must be emitted before the parent slot is finalized.
        int kids[8];
        for (int k = 0; k < 8; ++k) {
            kids[k] = node->children[k] ? flattenOctree(node->children[k].get(), nodes, triList,
                                                        depth + 1, maxDepth)
                                        : -1;
        }
        for (int k = 0; k < 8; ++k) n.child[k] = kids[k];
        nodes[id] = n;
    }
    return id;
}

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

// Bulk MT19937 word generator.  Produces exactly the same sequence as
// std::mt19937 (as used by the reference implementation), but emits whole
// blocks at a time so that the stream can be handed to the GPU in chunks.
class MT19937Bulk {
    static constexpr int NN = 624;
    static constexpr int MM = 397;
    uint32_t mt[NN];
    int idx;

    void twist() {
        constexpr uint32_t UPPER = 0x80000000u;
        constexpr uint32_t LOWER = 0x7fffffffu;
        for (int i = 0; i < NN - MM; ++i) {
            uint32_t y = (mt[i] & UPPER) | (mt[i + 1] & LOWER);
            mt[i] = mt[i + MM] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        for (int i = NN - MM; i < NN - 1; ++i) {
            uint32_t y = (mt[i] & UPPER) | (mt[i + 1] & LOWER);
            mt[i] = mt[i + MM - NN] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
        }
        uint32_t y = (mt[NN - 1] & UPPER) | (mt[0] & LOWER);
        mt[NN - 1] = mt[MM - 1] ^ (y >> 1) ^ ((y & 1u) ? 0x9908b0dfu : 0u);
    }

public:
    explicit MT19937Bulk(uint32_t seed = 42) {
        mt[0] = seed;
        for (int i = 1; i < NN; ++i) {
            mt[i] = 1812433253u * (mt[i - 1] ^ (mt[i - 1] >> 30)) + static_cast<uint32_t>(i);
        }
        idx = NN;
    }

    void generate(uint32_t* out, size_t count) {
        size_t done = 0;
        while (done < count) {
            if (idx == NN) {
                twist();
                idx = 0;
            }
            size_t take = std::min(count - done, static_cast<size_t>(NN - idx));
            const uint32_t* src = mt + idx;
            uint32_t* dst = out + done;
            for (size_t q = 0; q < take; ++q) {
                uint32_t y = src[q];
                y ^= y >> 11;
                y ^= (y << 7) & 0x9d2c5680u;
                y ^= (y << 15) & 0xefc60000u;
                y ^= y >> 18;
                dst[q] = y;
            }
            idx += static_cast<int>(take);
            done += take;
        }
    }
};

// Reproduce libstdc++'s std::uniform_real_distribution<float>(0,1) applied to a
// raw mt19937 word: generate_canonical<float,24> uses a single draw, scales it
// by 2^-32 (exact) and clamps the (possible) rounding to 1.0f down to the
// largest float below one.
__device__ __forceinline__ val_t canonicalFloat(uint32_t w) {
    val_t f = static_cast<val_t>(w) * 2.3283064365386963e-10f;  // 2^-32, exact
    return f >= 1.0f ? 0x1.fffffep-1f : f;
}

// ============================================================================
// Device geometry kernels helpers
// ============================================================================

// Same culling predicate as the reference computeKij().
__device__ __forceinline__ bool pairActive(const float4& ni, const float4& nj) {
    return !(ni.x * nj.x + ni.y * nj.y + ni.z * nj.z > 0.99f);
}

// Ray-Triangle Intersection (Moeller-Trumbore algorithm)
__device__ __forceinline__ val_t rayTriangleIntersectDev(const Vec3& orig, const Vec3& dir,
                                                         const Vec3& v0, const Vec3& v1,
                                                         const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (fabsf(det) < EPSILON) return __int_as_float(0x7f7fffff);  // FLT_MAX

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return __int_as_float(0x7f7fffff);

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return __int_as_float(0x7f7fffff);

    return e2.dot(qvec) * invDet;
}

// Segment/box overlap test, matching Octree::rayIntersectsBox.  The
// segment-dependent quantities (mid = p1 + d, d, |d|) are hoisted by the caller.
__device__ __forceinline__ bool segmentBoxOverlap(const float4& nc, const float4& nh,
                                                  const Vec3& mid, const Vec3& d, const Vec3& ad) {
    const Vec3 c(mid.x - nc.x, mid.y - nc.y, mid.z - nc.z);

    if (fabsf(c.x) > nh.x + ad.x) return false;
    if (fabsf(c.y) > nh.y + ad.y) return false;
    if (fabsf(c.z) > nh.z + ad.z) return false;

    // Same contraction pattern as the host compiler (see Vec3::dot).
    if (fabsf(__fmaf_rn(d.y, c.z, -__fmul_rn(d.z, c.y))) >
        __fmaf_rn(nh.y, ad.z, __fmul_rn(nh.z, ad.y)) + EPSILON) return false;
    if (fabsf(__fmaf_rn(d.z, c.x, -__fmul_rn(d.x, c.z))) >
        __fmaf_rn(nh.z, ad.x, __fmul_rn(nh.x, ad.z)) + EPSILON) return false;
    if (fabsf(__fmaf_rn(d.x, c.y, -__fmul_rn(d.y, c.x))) >
        __fmaf_rn(nh.x, ad.y, __fmul_rn(nh.y, ad.x)) + EPSILON) return false;

    return true;
}

// Maximum traversal stack depth (checked against the actual octree depth on the
// host after construction).
constexpr int TRAVERSAL_STACK = 72;

// Check if a ray between two triangles is blocked by any other triangle.
// Equivalent to isRayBlocked(): the octree only prunes candidates, and the
// answer ("is any triangle in the way") is independent of the visit order.
__device__ bool isRayBlockedDev(const Vec3& from, const Vec3& to, int srcTri, int dstTri,
                                const GNode* __restrict__ nodes,
                                const int* __restrict__ nodeTris,
                                const float4* __restrict__ triABC) {
    const Vec3 dir = to - from;
    const val_t rayLen = dir.norm();
    if (rayLen < EPSILON) return true;
    const Vec3 dirNorm = dir / rayLen;
    const val_t limit = rayLen - EPSILON;

    const Vec3 d = (to - from) * 0.5f;
    const Vec3 mid = from + d;
    const Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));

    int stack[TRAVERSAL_STACK];
    int sp = 0;
    stack[sp++] = 0;

    while (sp > 0) {
        const int ni = stack[--sp];
        const int triCount = nodes[ni].triCount;

        if (triCount > 0) {
            const int triStart = nodes[ni].triStart;
            for (int k = 0; k < triCount; ++k) {
                const int idx = nodeTris[triStart + k];
                if (idx == srcTri || idx == dstTri) continue;
                const float4 a = triABC[3 * idx + 0];
                const float4 b = triABC[3 * idx + 1];
                const float4 c = triABC[3 * idx + 2];
                const val_t dist = rayTriangleIntersectDev(from, dirNorm,
                                                           Vec3(a.x, a.y, a.z),
                                                           Vec3(b.x, b.y, b.z),
                                                           Vec3(c.x, c.y, c.z));
                if (dist > EPSILON && dist < limit) return true;  // Ray is blocked
            }
        } else {
            for (int k = 7; k >= 0; --k) {
                const int ci = nodes[ni].child[k];
                if (ci >= 0 && segmentBoxOverlap(nodes[ci].c, nodes[ci].h, mid, d, ad)) {
                    stack[sp++] = ci;
                }
            }
        }
    }
    return false;
}

// Compute the cosine of angle between vector and triangle normal
__device__ __forceinline__ val_t cosPhiDev(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// Kernels
// ============================================================================

// Block-wide exclusive prefix sum; also returns the total in *total.
__device__ __forceinline__ unsigned int blockExclusiveScan(unsigned int value,
                                                           unsigned int* scratch,
                                                           unsigned int* total) {
    const unsigned int tid = threadIdx.x;
    scratch[tid] = value;
    __syncthreads();
    for (unsigned int offset = 1; offset < blockDim.x; offset <<= 1) {
        unsigned int add = 0;
        if (tid >= offset) add = scratch[tid - offset];
        __syncthreads();
        scratch[tid] += add;
        __syncthreads();
    }
    *total = scratch[blockDim.x - 1];
    const unsigned int inclusive = scratch[tid];
    __syncthreads();
    return inclusive - value;
}

// Per-row scan of the non-culled (i.e. RNG-consuming) pairs: for every j-tile
// of a row, the number of active pairs of that row that precede the tile, plus
// the row totals.  This gives every pair its absolute position in the MT19937
// stream without materializing an N x N array.
__global__ void kernelRowScan(int N, int tilesPerRow, int tileSize,
                              const float4* __restrict__ triNrm,
                              unsigned int* __restrict__ tileBase,
                              unsigned int* __restrict__ rowActive) {
    const int i = blockIdx.x;
    const float4 ni = triNrm[i];

    __shared__ unsigned int s_scan[RED_BLOCK];
    __shared__ unsigned int s_running;
    if (threadIdx.x == 0) s_running = 0;
    __syncthreads();

    for (int t0 = 0; t0 < tilesPerRow; t0 += blockDim.x) {
        const int k = t0 + static_cast<int>(threadIdx.x);
        unsigned int count = 0;
        if (k < tilesPerRow) {
            const int jBegin = k * tileSize;
            const int jEnd = min(N, jBegin + tileSize);
            for (int j = jBegin; j < jEnd; ++j) {
                if (j != i && pairActive(ni, triNrm[j])) count++;
            }
        }
        unsigned int total = 0;
        const unsigned int offset = blockExclusiveScan(count, s_scan, &total);
        if (k < tilesPerRow) {
            tileBase[static_cast<size_t>(i) * tilesPerRow + k] = s_running + offset;
        }
        __syncthreads();
        if (threadIdx.x == 0) s_running += total;
        __syncthreads();
    }

    if (threadIdx.x == 0) rowActive[i] = s_running;
}

// Form factors: one block per (receiver triangle i, tile of emitters j), one
// thread per emitter j.  Each thread traces the 16 rays of its pair
// sequentially, so the per-pair accumulation order matches the reference.
__global__ __launch_bounds__(KIJ_BLOCK, 12) void kernelFormFactors(int N, int rowBegin, int tilesPerRow,
                                  const uint32_t* __restrict__ rng,
                                  unsigned long long rngPairBase,
                                  const unsigned long long* __restrict__ rowStart,
                                  const unsigned int* __restrict__ tileBase,
                                  const float4* __restrict__ triABC,
                                  const float4* __restrict__ triNrm,
                                  const GNode* __restrict__ nodes,
                                  const int* __restrict__ nodeTris,
                                  KijTau* __restrict__ kijTau) {
    const int i = rowBegin + static_cast<int>(blockIdx.y);
    const int tile = static_cast<int>(blockIdx.x);
    const int j = tile * KIJ_BLOCK + static_cast<int>(threadIdx.x);

    const float4 in4 = triNrm[i];

    float4 jn4 = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    bool active = false;
    if (j < N && j != i) {
        jn4 = triNrm[j];
        active = pairActive(in4, jn4);
    }

    // Rank of this pair within the random number stream.
    const unsigned int mask = __ballot_sync(0xffffffffu, active);
    __shared__ unsigned int s_warpCount[KIJ_BLOCK / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    if (lane == 0) s_warpCount[warp] = __popc(mask);
    __syncthreads();

    if (active) {
        unsigned int warpBase = 0;
        for (int w = 0; w < warp; ++w) warpBase += s_warpCount[w];

        const unsigned long long pairRank =
            rowStart[i] - rngPairBase +
            tileBase[static_cast<size_t>(i) * tilesPerRow + tile] + warpBase +
            __popc(mask & ((1u << lane) - 1u));
        const uint4* words = reinterpret_cast<const uint4*>(rng + pairRank * WORDS_PER_PAIR);

        const float4 ia = triABC[3 * i + 0];
        const float4 ib = triABC[3 * i + 1];
        const float4 ic = triABC[3 * i + 2];
        const Vec3 aI(ia.x, ia.y, ia.z);
        const Vec3 abI = Vec3(ib.x, ib.y, ib.z) - aI;
        const Vec3 acI = Vec3(ic.x, ic.y, ic.z) - aI;
        const Vec3 nI(in4.x, in4.y, in4.z);

        {
            const float4 ja = triABC[3 * j + 0];
            const float4 jb = triABC[3 * j + 1];
            const float4 jc = triABC[3 * j + 2];
            const Vec3 aJ(ja.x, ja.y, ja.z);
            const Vec3 abJ = Vec3(jb.x, jb.y, jb.z) - aJ;
            const Vec3 acJ = Vec3(jc.x, jc.y, jc.z) - aJ;
            const Vec3 nJ(jn4.x, jn4.y, jn4.z);

            val_t kij = ZERO;
            for (int r = 0; r < NUM_RAYS; ++r) {
                const uint4 w = words[r];

                val_t u = canonicalFloat(w.x);
                val_t v = canonicalFloat(w.y);
                if (u + v > 1.0f) { u = 1.0f - u; v = 1.0f - v; }
                const Vec3 pI = aI + abI * u + acI * v;

                val_t u2 = canonicalFloat(w.z);
                val_t v2 = canonicalFloat(w.w);
                if (u2 + v2 > 1.0f) { u2 = 1.0f - u2; v2 = 1.0f - v2; }
                const Vec3 pJ = aJ + abJ * u2 + acJ * v2;

                if (isRayBlockedDev(pI, pJ, i, j, nodes, nodeTris, triABC)) continue;

                const Vec3 dv = pJ - pI;
                const val_t distSqr = dv.squaredNorm();
                if (distSqr < EPSILON) continue;

                const val_t cosPhiI = cosPhiDev(dv, nI);
                const val_t cosPhiJ = cosPhiDev(-dv, nJ);
                if (cosPhiI <= ZERO || cosPhiJ <= ZERO) continue;

                kij += (cosPhiI * cosPhiJ) / (PI * distSqr);
            }

            // Transposed storage: entry [j][i] holds Kij(receiver i, emitter j)
            kijTau[static_cast<size_t>(j) * N + i].kij = kij * INV_NUM_RAYS;
        }
    }
}

// Time delays; stored transposed, interleaved with the form factors.
__global__ void kernelTau(int N, const float4* __restrict__ triCenter,
                          KijTau* __restrict__ kijTau) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int j = blockIdx.y;
    if (i >= N) return;

    int tau = 0;
    if (i != j) {
        const float4 ci = triCenter[i];
        const float4 cj = triCenter[j];
        const Vec3 delta(ci.x - cj.x, ci.y - cj.y, ci.z - cj.z);
        tau = static_cast<int>(ceilf(delta.norm() * INV_WAVE_SPEED));
    }
    kijTau[static_cast<size_t>(j) * N + i].tau = tau;
}

// One timestep of the radiosity update: one thread per receiver triangle,
// accumulating over the emitters in the original (ascending j) order.
__global__ void kernelSimStep(int N, int t,
                              const KijTau* __restrict__ kijTau,
                              const val_t* __restrict__ areas,
                              const val_t* __restrict__ rho,
                              const val_t* __restrict__ radE,
                              val_t* radB) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    // Emitters are accumulated in ascending j, exactly as in the reference, so
    // the floating point summation order is preserved.  The skip conditions are
    // turned into a predicated zero contribution (adding 0.0f is exact) so that
    // the independent loads of this loop can be pipelined by the compiler.
    val_t sumB = ZERO;
    size_t off = static_cast<size_t>(i);
    for (int j = 0; j < N; ++j, off += N) {
        const KijTau e = kijTau[off];
        const int tauij = e.tau;
        const val_t kij = e.kij;

        // Skip if the wave hasn't yet propagated from j to i
        const int srcTime = t - tauij;
        // Get radiosity from source triangle at time when emission occurred
        const val_t radJ = radB[static_cast<size_t>(srcTime > 0 ? srcTime : 0) * N + j];

        // Accumulate contribution: form factor * area * source radiosity
        // (kij is exactly zero for j == i and for culled pairs)
        const bool use = (srcTime >= 0) & (kij > ZERO) & (radJ > ZERO);
        sumB += use ? fminf(kij * __ldg(&areas[j]), ONE) * radJ : ZERO;
    }

    // Update radiosity: reflection + emission
    const size_t self = static_cast<size_t>(t) * N + i;
    radB[self] = rho[i] * sumB + radE[self];
}

// Cross-correlation: one block per triangle, one thread per lag.
__global__ void kernelDistances(int N, int T, int sourceIndex, int useShared,
                                const val_t* __restrict__ radB, val_t* __restrict__ distances) {
    extern __shared__ val_t s_rad[];
    val_t* sB = s_rad;          // radB[.][i]
    val_t* sS = s_rad + T;      // radB[.][sourceIndex]

    const int i = blockIdx.x;

    if (useShared) {
        for (int tt = threadIdx.x; tt < T; tt += blockDim.x) {
            sB[tt] = radB[static_cast<size_t>(tt) * N + i];
            sS[tt] = radB[static_cast<size_t>(tt) * N + sourceIndex];
        }
        __syncthreads();
    }

    // Same starting point as the reference scan: a lag is only accepted if it
    // strictly beats the best correlation so far, so lag 0 wins by default.
    val_t bestCorr = ZERO;
    int bestT = 0;

    for (int t = threadIdx.x; t < T; t += blockDim.x) {
        // The host compiler emits a rounded multiply followed by an add here
        // (the strided accesses keep it from contracting), so do the same.
        val_t sum = ZERO;
        if (useShared) {
            for (int tt = t; tt < T; ++tt) sum += __fmul_rn(sS[tt - t], sB[tt]);
        } else {
            for (int tt = t; tt < T; ++tt) {
                sum += __fmul_rn(radB[static_cast<size_t>(tt - t) * N + sourceIndex],
                                 radB[static_cast<size_t>(tt) * N + i]);
            }
        }
        if (sum > bestCorr || (sum == bestCorr && t < bestT)) {
            bestCorr = sum;
            bestT = t;
        }
    }

    // Block-wide "maximum, smallest lag wins ties" reduction: identical to the
    // reference scan `if (sum > maxCorr)` over ascending t.
    __shared__ val_t s_val[DIST_BLOCK];
    __shared__ int s_idx[DIST_BLOCK];
    s_val[threadIdx.x] = bestCorr;
    s_idx[threadIdx.x] = bestT;
    __syncthreads();
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const val_t ov = s_val[threadIdx.x + stride];
            const int oi = s_idx[threadIdx.x + stride];
            if (ov > s_val[threadIdx.x] || (ov == s_val[threadIdx.x] && oi < s_idx[threadIdx.x])) {
                s_val[threadIdx.x] = ov;
                s_idx[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        distances[i] = WAVE_SPEED * static_cast<val_t>(s_idx[0]);
    }
}

// Count of form factors above EPSILON (validation statistic).
__global__ void kernelCountNonZero(size_t n, const KijTau* __restrict__ data,
                                   unsigned long long* __restrict__ result) {
    if (n == 0) return;
    size_t stride = static_cast<size_t>(gridDim.x) * blockDim.x;
    unsigned long long count = 0;
    for (size_t p = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x; p < n; p += stride) {
        if (data[p].kij > EPSILON) count++;
    }
    for (int off = 16; off > 0; off >>= 1) count += __shfl_down_sync(0xffffffffu, count, off);
    if ((threadIdx.x & 31) == 0) atomicAdd(result, count);
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
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix, host mirror)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure (host, build only)

    size_t sourceIndex;
    unsigned long long nonZeroKij = 0;

    // Device state
    float4* d_triABC = nullptr;     // 3 vertices per triangle (interleaved)
    float4* d_triNrm = nullptr;
    float4* d_triCenter = nullptr;
    val_t* d_areas = nullptr;
    val_t* d_rho = nullptr;
    KijTau* d_kijTau = nullptr;     // Form factors + delays (N x N, transposed)
    val_t* d_radE = nullptr;        // Emission radiosity (T x N)
    val_t* d_radB = nullptr;        // Reflected radiosity (T x N)
    val_t* d_distances = nullptr;
    GNode* d_nodes = nullptr;
    int* d_nodeTris = nullptr;
    unsigned long long* d_rowStart = nullptr;
    unsigned int* d_rowActive = nullptr;
    unsigned int* d_tileBase = nullptr;   // active pairs per row before each j-tile
    int tilesPerRow = 0;

    // Double-buffered staging area for the MT19937 stream: the host generates
    // chunk c while chunk c-1 is transferred and chunk c-2 is being consumed.
    size_t rngCapacity = 0;
    int numRngBuffers = 1;
    uint32_t* h_rng[2] = {nullptr, nullptr};
    uint32_t* d_rng[2] = {nullptr, nullptr};
    cudaStream_t copyStream = nullptr, compStream = nullptr;
    cudaEvent_t copyDone[2] = {nullptr, nullptr};
    cudaEvent_t kernDone[2] = {nullptr, nullptr};

    std::vector<unsigned long long> rowStart;  // prefix sum of active pairs per row (N+1)

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

    const size_t N = state.numTriangles;

    // Initialize areas
    state.areas.resize(N);
    for (size_t i = 0; i < N; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.assign(N, reflectivity);

    state.radB.assign(timesteps * N, ZERO);
    state.distances.assign(N, ZERO);

    // Set source emission (active for first half of timesteps)
    std::vector<val_t> radE(timesteps * N, ZERO);
    size_t timeOn = 0;
    size_t timeOff = timesteps / 2;
    for (size_t t = timeOn; t < timeOff; ++t) {
        radE[state.idxTN(t, state.sourceIndex)] = 1.0f;
    }

    // ---- Device setup -----------------------------------------------------
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));
    if (deviceCount == 0) {
        fprintf(stderr, "No CUDA capable device found\n");
        exit(EXIT_FAILURE);
    }
    CUDA_CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("CUDA device: %s (%d SMs)\n", prop.name, prop.multiProcessorCount);

    // Flatten mesh for the GPU
    std::vector<float4> triABC(3 * N), triNrm(N), triCenter(N);
    for (size_t i = 0; i < N; ++i) {
        const Triangle& tri = state.triangles[i];
        triABC[3 * i + 0] = make_float4(tri.a.x, tri.a.y, tri.a.z, 0.0f);
        triABC[3 * i + 1] = make_float4(tri.b.x, tri.b.y, tri.b.z, 0.0f);
        triABC[3 * i + 2] = make_float4(tri.c.x, tri.c.y, tri.c.z, 0.0f);
        const Vec3 n = tri.normal();
        triNrm[i] = make_float4(n.x, n.y, n.z, 0.0f);
        const Vec3 c = tri.center();
        triCenter[i] = make_float4(c.x, c.y, c.z, 0.0f);
    }

    // Flatten the octree
    std::vector<GNode> nodes;
    std::vector<int> nodeTris;
    nodes.reserve(2 * N);
    nodeTris.reserve(8 * N);
    int maxDepth = 0;
    flattenOctree(&state.octree, nodes, nodeTris, 0, maxDepth);
    if (7 * (maxDepth + 1) + 1 > TRAVERSAL_STACK) {
        fprintf(stderr, "Octree too deep (%d) for the traversal stack\n", maxDepth);
        exit(EXIT_FAILURE);
    }
    printf("Octree: %zu nodes, depth %d, %zu triangle references\n",
           nodes.size(), maxDepth, nodeTris.size());

    const size_t matElems = N * N;
    CUDA_CHECK(cudaMalloc(&state.d_triABC, 3 * N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.d_triNrm, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.d_triCenter, N * sizeof(float4)));
    CUDA_CHECK(cudaMalloc(&state.d_areas, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_rho, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_kijTau, matElems * sizeof(KijTau)));
    CUDA_CHECK(cudaMalloc(&state.d_radE, timesteps * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_radB, timesteps * N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_distances, N * sizeof(val_t)));
    CUDA_CHECK(cudaMalloc(&state.d_nodes, nodes.size() * sizeof(GNode)));
    CUDA_CHECK(cudaMalloc(&state.d_nodeTris, nodeTris.size() * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&state.d_rowStart, (N + 1) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&state.d_rowActive, N * sizeof(unsigned int)));
    state.tilesPerRow = static_cast<int>((N + KIJ_BLOCK - 1) / KIJ_BLOCK);
    CUDA_CHECK(cudaMalloc(&state.d_tileBase,
                          N * static_cast<size_t>(state.tilesPerRow) * sizeof(unsigned int)));

    CUDA_CHECK(cudaMemcpy(state.d_triABC, triABC.data(), 3 * N * sizeof(float4),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_triNrm, triNrm.data(), N * sizeof(float4),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_triCenter, triCenter.data(), N * sizeof(float4),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_areas, state.areas.data(), N * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_rho, state.rho.data(), N * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_radE, radE.data(), timesteps * N * sizeof(val_t),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodes, nodes.data(), nodes.size() * sizeof(GNode),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(state.d_nodeTris, nodeTris.data(), nodeTris.size() * sizeof(int),
                          cudaMemcpyHostToDevice));

    // Matrices start out zeroed, as in the reference implementation.
    CUDA_CHECK(cudaMemset(state.d_kijTau, 0, matElems * sizeof(KijTau)));
    CUDA_CHECK(cudaMemset(state.d_radB, 0, timesteps * N * sizeof(val_t)));

    // Staging buffers for the random number stream.  At most 64 words are
    // consumed per triangle pair, which bounds the total; one chunk must always
    // be able to hold a whole row.
    const size_t worstCaseWords = N * (N - 1) * static_cast<size_t>(WORDS_PER_PAIR);
    state.rngCapacity = std::min(worstCaseWords, RNG_CHUNK_WORDS);
    state.rngCapacity = std::max(state.rngCapacity, N * static_cast<size_t>(WORDS_PER_PAIR));
    state.numRngBuffers = worstCaseWords > state.rngCapacity ? 2 : 1;
    CUDA_CHECK(cudaStreamCreate(&state.copyStream));
    CUDA_CHECK(cudaStreamCreate(&state.compStream));
    for (int b = 0; b < state.numRngBuffers; ++b) {
        CUDA_CHECK(cudaMallocHost(&state.h_rng[b], state.rngCapacity * sizeof(uint32_t)));
        CUDA_CHECK(cudaMalloc(&state.d_rng[b], state.rngCapacity * sizeof(uint32_t)));
        CUDA_CHECK(cudaEventCreateWithFlags(&state.copyDone[b], cudaEventDisableTiming));
        CUDA_CHECK(cudaEventCreateWithFlags(&state.kernDone[b], cudaEventDisableTiming));
    }

    // Force module load / JIT of every kernel before the timed phases start.
    // All warm-up launches are either empty or idempotent with respect to the
    // real launches that follow.
    kernelRowScan<<<1, RED_BLOCK>>>(0, 0, KIJ_BLOCK, state.d_triNrm, state.d_tileBase,
                                    state.d_rowActive);
    kernelFormFactors<<<dim3(1, 1), KIJ_BLOCK>>>(0, 0, state.tilesPerRow, state.d_rng[0], 0,
                                                 state.d_rowStart, state.d_tileBase,
                                                 state.d_triABC, state.d_triNrm, state.d_nodes,
                                                 state.d_nodeTris, state.d_kijTau);
    kernelTau<<<dim3(1, 1), dim3(256)>>>(0, state.d_triCenter, state.d_kijTau);
    kernelSimStep<<<1, SIM_BLOCK>>>(0, 0, state.d_kijTau, state.d_areas,
                                    state.d_rho, state.d_radE, state.d_radB);
    kernelDistances<<<1, DIST_BLOCK>>>(0, 0, 0, 0, state.d_radB, state.d_distances);
    kernelCountNonZero<<<1, RED_BLOCK>>>(0, state.d_kijTau, nullptr);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemset(state.d_distances, 0, N * sizeof(val_t)));
}

void releaseSimulation(SimulationState& state) {
    cudaFree(state.d_triABC);
    cudaFree(state.d_triNrm);
    cudaFree(state.d_triCenter);
    cudaFree(state.d_areas);
    cudaFree(state.d_rho);
    cudaFree(state.d_kijTau);
    cudaFree(state.d_radE);
    cudaFree(state.d_radB);
    cudaFree(state.d_distances);
    cudaFree(state.d_nodes);
    cudaFree(state.d_nodeTris);
    cudaFree(state.d_rowStart);
    cudaFree(state.d_rowActive);
    cudaFree(state.d_tileBase);
    for (int b = 0; b < state.numRngBuffers; ++b) {
        cudaFreeHost(state.h_rng[b]);
        cudaFree(state.d_rng[b]);
        cudaEventDestroy(state.copyDone[b]);
        cudaEventDestroy(state.kernDone[b]);
    }
    cudaStreamDestroy(state.copyStream);
    cudaStreamDestroy(state.compStream);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij)...\n");
    const int N = static_cast<int>(state.numTriangles);

    // ---- 1. Position of every pair in the MT19937 stream ------------------
    kernelRowScan<<<N, RED_BLOCK>>>(N, state.tilesPerRow, KIJ_BLOCK, state.d_triNrm,
                                    state.d_tileBase, state.d_rowActive);
    CUDA_CHECK(cudaGetLastError());

    std::vector<unsigned int> rowActive(N);
    CUDA_CHECK(cudaMemcpy(rowActive.data(), state.d_rowActive, N * sizeof(unsigned int),
                          cudaMemcpyDeviceToHost));

    state.rowStart.assign(N + 1, 0);
    unsigned long long maxRowPairs = 0;
    for (int i = 0; i < N; ++i) {
        state.rowStart[i + 1] = state.rowStart[i] + rowActive[i];
        maxRowPairs = std::max<unsigned long long>(maxRowPairs, rowActive[i]);
    }
    CUDA_CHECK(cudaMemcpy(state.d_rowStart, state.rowStart.data(),
                          (N + 1) * sizeof(unsigned long long), cudaMemcpyHostToDevice));

    // ---- 2. Split the rows into chunks of random numbers ------------------
    const size_t capacity = state.rngCapacity;
    if (static_cast<size_t>(maxRowPairs) * WORDS_PER_PAIR > capacity) {
        fprintf(stderr, "Random number staging buffer too small\n");
        exit(EXIT_FAILURE);
    }

    std::vector<int> chunkEnd;
    for (int r = 0; r < N;) {
        int r1 = r;
        while (r1 < N &&
               static_cast<size_t>(state.rowStart[r1 + 1] - state.rowStart[r]) * WORDS_PER_PAIR
                   <= capacity) {
            ++r1;
        }
        if (r1 == r) ++r1;  // cannot happen: capacity >= largest row
        chunkEnd.push_back(r1);
        r = r1;
    }

    const int numBuffers = chunkEnd.size() > 1 ? state.numRngBuffers : 1;

    // ---- 3. Stream the MT19937 output to the GPU, chunk by chunk ----------
    // Three-stage software pipeline: the host generates chunk c while chunk c-1
    // is copied on copyStream and chunk c-2 is consumed on compStream.
    MT19937Bulk rng(42);
    int rowBegin = 0;
    for (size_t c = 0; c < chunkEnd.size(); ++c) {
        const int rowEnd = chunkEnd[c];
        const int b = static_cast<int>(c) % numBuffers;
        const size_t words =
            static_cast<size_t>(state.rowStart[rowEnd] - state.rowStart[rowBegin]) * WORDS_PER_PAIR;

        // Wait until the host staging buffer is no longer being transferred.
        if (static_cast<int>(c) >= numBuffers) {
            CUDA_CHECK(cudaEventSynchronize(state.copyDone[b]));
        }

        rng.generate(state.h_rng[b], words);

        // The device buffer may only be overwritten once its consumer is done.
        if (static_cast<int>(c) >= numBuffers) {
            CUDA_CHECK(cudaStreamWaitEvent(state.copyStream, state.kernDone[b], 0));
        }
        CUDA_CHECK(cudaMemcpyAsync(state.d_rng[b], state.h_rng[b], words * sizeof(uint32_t),
                                   cudaMemcpyHostToDevice, state.copyStream));
        CUDA_CHECK(cudaEventRecord(state.copyDone[b], state.copyStream));

        CUDA_CHECK(cudaStreamWaitEvent(state.compStream, state.copyDone[b], 0));
        kernelFormFactors<<<dim3(state.tilesPerRow, rowEnd - rowBegin), KIJ_BLOCK, 0,
                            state.compStream>>>(
            N, rowBegin, state.tilesPerRow, state.d_rng[b], state.rowStart[rowBegin],
            state.d_rowStart, state.d_tileBase, state.d_triABC, state.d_triNrm, state.d_nodes,
            state.d_nodeTris, state.d_kijTau);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(state.kernDone[b], state.compStream));

        for (int i = rowBegin; i < rowEnd; ++i) {
            if ((i + 1) % 100 == 0 || i + 1 == N) {
                printf("  Progress: %zu/%zu triangles\n", static_cast<size_t>(i + 1),
                       state.numTriangles);
            }
        }
        rowBegin = rowEnd;
    }

    CUDA_CHECK(cudaDeviceSynchronize());
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau)...\n");
    const int N = static_cast<int>(state.numTriangles);

    dim3 block(256);
    dim3 grid((N + block.x - 1) / block.x, N);
    kernelTau<<<grid, block>>>(N, state.d_triCenter, state.d_kijTau);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// Simulation Phase (Wave Propagation)
// ============================================================================

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);
    const int grid = (N + SIM_BLOCK - 1) / SIM_BLOCK;

    for (int t = 0; t < T; ++t) {
        kernelSimStep<<<grid, SIM_BLOCK>>>(N, t, state.d_kijTau, state.d_areas,
                                           state.d_rho, state.d_radE, state.d_radB);
        CUDA_CHECK(cudaGetLastError());

        if ((t + 1) % 10 == 0 || t + 1 == T) {
            printf("  Timestep %zu/%zu\n", static_cast<size_t>(t + 1), state.numTimesteps);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.radB.data(), state.d_radB,
                          state.radB.size() * sizeof(val_t), cudaMemcpyDeviceToHost));
}

// ============================================================================
// Distance Computation (Cross-Correlation)
// ============================================================================

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation...\n");
    const int N = static_cast<int>(state.numTriangles);
    const int T = static_cast<int>(state.numTimesteps);

    const size_t sharedBytes = 2 * static_cast<size_t>(T) * sizeof(val_t);
    const int useShared = sharedBytes <= 40960 ? 1 : 0;

    kernelDistances<<<N, DIST_BLOCK, useShared ? sharedBytes : 0>>>(
        N, T, static_cast<int>(state.sourceIndex), useShared, state.d_radB, state.d_distances);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(state.distances.data(), state.d_distances, N * sizeof(val_t),
                          cudaMemcpyDeviceToHost));
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
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
    const size_t matElems = state.numTriangles * state.numTriangles;
    unsigned long long* d_count = nullptr;
    CUDA_CHECK(cudaMalloc(&d_count, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_count, 0, sizeof(unsigned long long)));
    kernelCountNonZero<<<1024, RED_BLOCK>>>(matElems, state.d_kijTau, d_count);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaMemcpy(&state.nonZeroKij, d_count, sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_count));

    const int nonZeroKij = static_cast<int>(state.nonZeroKij);
    printf("  Non-zero form factors: %d/%zu (%.2f%%)\n",
           nonZeroKij, matElems, 100.0f * nonZeroKij / static_cast<val_t>(matElems));

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
            releaseSimulation(state);
            return 1;
        }
    }

    releaseSimulation(state);
    return 0;
}
