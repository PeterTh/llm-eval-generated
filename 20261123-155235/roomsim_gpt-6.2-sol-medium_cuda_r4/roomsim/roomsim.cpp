/**
 * Room Response Simulation Benchmark
 *
 * This is a CUDA implementation of room impulse response
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
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <vector>

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

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    __host__ __device__ explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
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

// The depth-first octree layout stores the next node to visit when a subtree
// can be skipped. The traversal needs no per-ray stack.
struct DeviceNode {
    Vec3 center, halfExtent;
    idx_t escape;
    idx_t firstTriangle;
    idx_t triangleCount;
};

void flattenOctree(const Octree& tree, std::vector<DeviceNode>& nodes,
                    std::vector<idx_t>& indices) {
    const auto node = static_cast<idx_t>(nodes.size());
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<idx_t>(indices.size()),
                     static_cast<idx_t>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[node].escape = static_cast<idx_t>(nodes.size());
}

struct DeviceTriangle {
    Vec3 a, ab, ac;
    Vec3 normal, center;
    val_t area;
};

__device__ val_t deviceDot(const Vec3& a, const Vec3& b) {
    // Use the same contraction order as the benchmark's optimized host build.
    return fmaf(a.z, b.z, fmaf(a.x, b.x, a.y * b.y));
}

__device__ bool intersectsBox(const DeviceNode& node, const Vec3& midpoint,
                              const Vec3& d, const Vec3& ad) {
    const Vec3 c = midpoint - node.center;
    const Vec3 h = node.halfExtent;
    if (fabsf(c.x) > h.x + ad.x || fabsf(c.y) > h.y + ad.y || fabsf(c.z) > h.z + ad.z) {
        return false;
    }

    return fabsf(d.y * c.z - d.z * c.y) <= h.y * ad.z + h.z * ad.y + EPSILON &&
           fabsf(d.z * c.x - d.x * c.z) <= h.z * ad.x + h.x * ad.z + EPSILON &&
           fabsf(d.x * c.y - d.y * c.x) <= h.x * ad.y + h.y * ad.x + EPSILON;
}

__device__ bool intersectsTriangle(const DeviceTriangle& tri, const Vec3& origin,
                                   const Vec3& direction, val_t rayLength) {
    const Vec3 p = direction.cross(tri.ac);
    const val_t det = deviceDot(tri.ab, p);
    if (fabsf(det) < EPSILON) return false;

    const val_t invDet = 1.0f / det;
    const Vec3 t = origin - tri.a;
    const val_t u = deviceDot(t, p) * invDet;
    if (u < ZERO || u > ONE) return false;

    const Vec3 q = t.cross(tri.ab);
    const val_t v = deviceDot(direction, q) * invDet;
    if (v < ZERO || u + v > ONE) return false;

    const val_t distance = deviceDot(tri.ac, q) * invDet;
    return distance > EPSILON && distance < rayLength - EPSILON;
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& ray, val_t length,
                             idx_t i, idx_t j, const DeviceTriangle* triangles,
                             const DeviceNode* nodes, const idx_t* indices) {
    const Vec3 halfRay = ray * 0.5f;
    const Vec3 midpoint = from + halfRay;
    const Vec3 ad(fabsf(halfRay.x), fabsf(halfRay.y), fabsf(halfRay.z));
    const Vec3 direction = ray / length;

    const idx_t end = nodes[0].escape;
    for (idx_t k = 0; k < end;) {
        const DeviceNode node = nodes[k];
        if (k != 0 && !intersectsBox(node, midpoint, halfRay, ad)) {
            k = node.escape;
            continue;
        }

        for (idx_t m = 0; m < node.triangleCount; ++m) {
            const idx_t id = indices[node.firstTriangle + m];
            if (id != i && id != j && intersectsTriangle(triangles[id], from, direction, length)) {
                return true;
            }
        }
        ++k;
    }
    return false;
}

// A checkpoint holds a block of untempered MT19937 words and the offset of
// the next draw. The host advances only the recurrence, without producing
// random floats. Each CUDA block resumes its checkpoint independently.
constexpr int MT_WORDS = 624;
constexpr int MT_PERIOD_OFFSET = 397;
constexpr int MT_SPLIT = MT_WORDS - MT_PERIOD_OFFSET;

__host__ __device__ uint32_t twistWord(uint32_t a, uint32_t b, uint32_t c) {
    const uint32_t joined = (a & 0x80000000u) | (b & 0x7fffffffu);
    return c ^ (joined >> 1) ^ ((0u - (joined & 1u)) & 0x9908b0dfu);
}

struct RandomCheckpoint {
    uint32_t words[MT_WORDS];
    int cursor;
};

class RandomSequence {
    RandomCheckpoint state{};

    void twist() {
        auto& words = state.words;
        for (int k = 0; k < MT_SPLIT; ++k) {
            words[k] = twistWord(words[k], words[k + 1], words[k + MT_PERIOD_OFFSET]);
        }
        for (int k = MT_SPLIT; k < MT_WORDS - 1; ++k) {
            words[k] = twistWord(words[k], words[k + 1], words[k - MT_SPLIT]);
        }
        words[MT_WORDS - 1] = twistWord(words[MT_WORDS - 1], words[0], words[MT_PERIOD_OFFSET - 1]);
    }

public:
    explicit RandomSequence(uint32_t seed) {
        state.words[0] = seed;
        for (int k = 1; k < MT_WORDS; ++k) {
            const auto prev = state.words[k - 1];
            state.words[k] = 1812433253u * (prev ^ (prev >> 30)) + k;
        }
        twist();
    }

    void reserve(RandomCheckpoint& checkpoint, int draws) {
        if (draws != 0) checkpoint = state;
        state.cursor += draws;
        while (state.cursor >= MT_WORDS) {
            twist();
            state.cursor -= MT_WORDS;
        }
    }
};

constexpr int BLOCK_SIZE = 128;
constexpr int PAIRS_PER_BLOCK = BLOCK_SIZE;
constexpr int PAIRS_PER_WAVE = BLOCK_SIZE / NUM_RAYS;
constexpr int DRAWS_PER_PAIR = NUM_RAYS * 4;
constexpr int DRAWS_PER_WAVE = PAIRS_PER_WAVE * DRAWS_PER_PAIR;
static_assert(DRAWS_PER_WAVE <= MT_WORDS);

__device__ void twistBlock(const uint32_t* oldWords, uint32_t* newWords) {
    // A later segment reads words already produced by the previous segment.
    for (int k = threadIdx.x; k < MT_SPLIT; k += BLOCK_SIZE) {
        newWords[k] = twistWord(oldWords[k], oldWords[k + 1], oldWords[k + MT_PERIOD_OFFSET]);
    }
    __syncthreads();
    for (int k = MT_SPLIT + threadIdx.x; k < 2 * MT_SPLIT; k += BLOCK_SIZE) {
        newWords[k] = twistWord(oldWords[k], oldWords[k + 1], newWords[k - MT_SPLIT]);
    }
    __syncthreads();
    for (int k = 2 * MT_SPLIT + threadIdx.x; k < MT_WORDS - 1; k += BLOCK_SIZE) {
        newWords[k] = twistWord(oldWords[k], oldWords[k + 1], newWords[k - MT_SPLIT]);
    }
    if (threadIdx.x == 0) {
        newWords[MT_WORDS - 1] = twistWord(oldWords[MT_WORDS - 1], newWords[0],
                                           newWords[MT_PERIOD_OFFSET - 1]);
    }
    __syncthreads();
}

__device__ val_t randomFloat(uint32_t word) {
    word ^= word >> 11;
    word ^= (word << 7) & 0x9d2c5680u;
    word ^= (word << 15) & 0xefc60000u;
    word ^= word >> 18;
    // Match the one-draw uniform_real_distribution<float>, including the
    // clamp for integers that round to 2^32 during conversion.
    return fminf(static_cast<val_t>(word) * 0x1p-32f, 0x1.fffffep-1f);
}

__device__ Vec3 randomPoint(const DeviceTriangle& tri, val_t u, val_t v) {
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return {fmaf(tri.ac.x, v, fmaf(tri.ab.x, u, tri.a.x)),
            fmaf(tri.ac.y, v, fmaf(tri.ab.y, u, tri.a.y)),
            fmaf(tri.ac.z, v, fmaf(tri.ab.z, u, tri.a.z))};
}

__device__ val_t sampleFormFactor(idx_t i, idx_t j, const DeviceTriangle* triangles,
                                  const DeviceNode* nodes, const idx_t* indices,
                                  const val_t (&draws)[4]) {
    const DeviceTriangle triI = triangles[i];
    const DeviceTriangle triJ = triangles[j];
    const Vec3 from = randomPoint(triI, draws[0], draws[1]);
    const Vec3 ray = randomPoint(triJ, draws[2], draws[3]) - from;
    const val_t distSqr = deviceDot(ray, ray);
    if (distSqr < EPSILON) return ZERO;

    const val_t length = sqrtf(distSqr);
    const val_t cosI = deviceDot(ray, triI.normal) / length;
    const val_t cosJ = deviceDot(-ray, triJ.normal) / length;
    if (cosI <= ZERO || cosJ <= ZERO || isRayBlocked(from, ray, length, i, j, triangles, nodes, indices)) {
        return ZERO;
    }
    return (cosI * cosJ) / (PI * distSqr);
}

__device__ bool samplePair(idx_t i, idx_t j, const DeviceTriangle* triangles) {
    return i != j && deviceDot(triangles[i].normal, triangles[j].normal) <= 0.99f;
}

__global__ void countPairs(const DeviceTriangle* triangles, idx_t n, idx_t blocksPerRow,
                           int* counts) {
    const idx_t i = blockIdx.x / blocksPerRow;
    const idx_t j = (blockIdx.x % blocksPerRow) * PAIRS_PER_BLOCK + threadIdx.x;
    const int count = __syncthreads_count(j < n && samplePair(i, j, triangles));
    if (threadIdx.x == 0) counts[blockIdx.x] = count;
}

__global__ void formFactors(const DeviceTriangle* triangles, const DeviceNode* nodes,
                            const idx_t* indices, idx_t n, idx_t blocksPerRow, idx_t firstBlock,
                            const int* counts, const RandomCheckpoint* seeds,
                            val_t* weights, uint8_t* delays, unsigned long long* nonZero) {
    const idx_t tile = firstBlock + blockIdx.x;
    const idx_t i = tile / blocksPerRow;
    const idx_t firstJ = (tile % blocksPerRow) * PAIRS_PER_BLOCK;
    const idx_t j = firstJ + threadIdx.x;
    const size_t row = static_cast<size_t>(i) * n;
    const bool active = j < n && samplePair(i, j, triangles);
    if (j < n && !active) delays[row + j] = 0;

    const int count = counts[tile];
    if (count == 0) return;

    __shared__ int warpCounts[BLOCK_SIZE / 32];
    __shared__ int selected[PAIRS_PER_BLOCK];
    __shared__ uint32_t words[2][MT_WORDS];
    __shared__ val_t samples[BLOCK_SIZE];

    const int lane = threadIdx.x % 32;
    const int warp = threadIdx.x / 32;
    const unsigned mask = __ballot_sync(0xffffffffu, active);
    if (lane == 0) warpCounts[warp] = __popc(mask);

    const RandomCheckpoint& seed = seeds[blockIdx.x];
    for (int k = threadIdx.x; k < MT_WORDS; k += BLOCK_SIZE) {
        words[0][k] = seed.words[k];
    }
    __syncthreads();

    if (active) {
        int rank = __popc(mask & ((1u << lane) - 1u));
        for (int w = 0; w < warp; ++w) rank += warpCounts[w];
        selected[rank] = threadIdx.x;
    }
    __syncthreads();

    int current = 0;
    int cursor = seed.cursor;
    int countAboveEpsilon = 0;
    for (int base = 0; base < count; base += PAIRS_PER_WAVE) {
        const int activePairs = min(PAIRS_PER_WAVE, count - base);
        const bool advance = cursor + activePairs * DRAWS_PER_PAIR >= MT_WORDS;
        if (advance) twistBlock(words[current], words[1 - current]);

        const int slot = threadIdx.x / NUM_RAYS;
        if (slot < activePairs) {
            val_t draws[4];
            const int firstDraw = cursor + 4 * threadIdx.x;
            for (int k = 0; k < 4; ++k) {
                const int draw = firstDraw + k;
                draws[k] = randomFloat(draw < MT_WORDS ? words[current][draw]
                                                       : words[1 - current][draw - MT_WORDS]);
            }
            const idx_t sender = firstJ + selected[base + slot];
            samples[threadIdx.x] = sampleFormFactor(i, sender, triangles, nodes, indices, draws);

            // Repeat the CPU's ordered accumulation of the sixteen rays.
            __syncwarp(0xffffu << (lane & 16));
            if (threadIdx.x % NUM_RAYS == 0) {
                val_t sum = ZERO;
                #pragma unroll
                for (int r = 0; r < NUM_RAYS; ++r) sum += samples[threadIdx.x + r];
                const val_t kij = sum * INV_NUM_RAYS;
                const Vec3 d = triangles[i].center - triangles[sender].center;
                const int tau = __float2int_ru(sqrtf(deviceDot(d, d)) * INV_WAVE_SPEED);
                // The radius-10 mesh has delays in [1, 40] off the diagonal.
                delays[row + sender] = kij > ZERO ? static_cast<uint8_t>(tau) : 0;
                weights[row + sender] = fminf(kij * triangles[sender].area, ONE);
                countAboveEpsilon += kij > EPSILON;
            }
        }

        __syncthreads();
        cursor += activePairs * DRAWS_PER_PAIR;
        if (advance) {
            cursor -= MT_WORDS;
            current = 1 - current;
        }
    }

    if (nonZero && threadIdx.x % NUM_RAYS == 0 && countAboveEpsilon != 0) {
        atomicAdd(nonZero, static_cast<unsigned long long>(countAboveEpsilon));
    }
}

__global__ void propagate(const val_t* weights, const uint8_t* delays, val_t* radiosity,
                          idx_t n, idx_t source, int timestep, int timeOff, val_t rho) {
    const idx_t i = blockIdx.x;
    const size_t row = static_cast<size_t>(i) * n;
    __shared__ val_t terms[BLOCK_SIZE];
    val_t sum = ZERO;

    // At t = 0 only the emission can contribute.
    for (idx_t base = 0; timestep != 0 && base < n; base += BLOCK_SIZE) {
        const idx_t j = base + threadIdx.x;
        val_t contribution = ZERO;
        if (j < n) {
            const int delay = delays[row + j];
            if (delay != 0 && delay <= timestep) {
                const val_t prior = radiosity[static_cast<size_t>(timestep - delay) * n + j];
                if (!(prior <= ZERO)) contribution = weights[row + j] * prior;
            }
        }
        terms[threadIdx.x] = contribution;
        __syncthreads();
        if (threadIdx.x == 0) {
            const int count = min(static_cast<idx_t>(BLOCK_SIZE), n - base);
            #pragma unroll 8
            for (int k = 0; k < count; ++k) sum += terms[k];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        const val_t emission = i == source && timestep < timeOff ? ONE : ZERO;
        radiosity[static_cast<size_t>(timestep) * n + i] = fmaf(rho, sum, emission);
    }
}

constexpr int CORR_TRIANGLES = 8;
constexpr int CORR_LAGS = 16;

__global__ void correlate(const val_t* radiosity, val_t* distances, uint8_t* energy,
                          idx_t n, idx_t source, int timesteps) {
    const idx_t i = blockIdx.x * CORR_TRIANGLES + threadIdx.x;
    const int lane = threadIdx.y;
    __shared__ val_t scores[CORR_LAGS][CORR_TRIANGLES];
    __shared__ int lags[CORR_LAGS][CORR_TRIANGLES];
    val_t best = ZERO;
    int bestLag = 0;
    bool received = false;

    if (i < n) {
        for (int lag = lane; lag < timesteps; lag += CORR_LAGS) {
            val_t sum = ZERO;
            for (int t = lag; t < timesteps; ++t) {
                const size_t base = static_cast<size_t>(t) * n;
                const val_t signal = radiosity[base + i];
                const val_t reference = radiosity[static_cast<size_t>(t - lag) * n + source];
                sum = __fadd_rn(sum, __fmul_rn(signal, reference));
                if (energy && lag == 0) received |= signal > EPSILON;
            }
            if (sum > best) {
                best = sum;
                bestLag = lag;
            }
        }
    }

    scores[lane][threadIdx.x] = best;
    lags[lane][threadIdx.x] = bestLag;
    __syncthreads();

    if (lane == 0 && i < n) {
        for (int k = 1; k < CORR_LAGS; ++k) {
            const val_t score = scores[k][threadIdx.x];
            const int lag = lags[k][threadIdx.x];
            if (score > best || (score == best && lag < bestLag)) {
                best = score;
                bestLag = lag;
            }
        }
        distances[i] = WAVE_SPEED * static_cast<val_t>(bestLag);
        if (energy) energy[i] = received;
    }
}

void checkCuda(cudaError_t status, const char* operation) {
    if (status != cudaSuccess) {
        throw std::runtime_error(std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

template <typename T>
class DeviceBuffer {
    T* data = nullptr;

public:
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }

    void allocate(size_t count, bool managed = false) {
        if (count == 0) return;
        if (managed) {
            checkCuda(cudaMallocManaged(&data, count * sizeof(T)), "Allocating CUDA managed memory");
        } else {
            checkCuda(cudaMalloc(&data, count * sizeof(T)), "Allocating CUDA device memory");
        }
    }

    void upload(const std::vector<T>& values) {
        allocate(values.size());
        if (!values.empty()) {
            checkCuda(cudaMemcpy(data, values.data(), values.size() * sizeof(T),
                                 cudaMemcpyHostToDevice), "Uploading simulation data");
        }
    }

    void download(T* output, size_t count) const {
        checkCuda(cudaMemcpy(output, data, count * sizeof(T), cudaMemcpyDeviceToHost),
                  "Downloading simulation data");
    }

    T* get() const { return data; }
};

class SeedBatch {
    cudaStream_t stream = nullptr;
    RandomCheckpoint* hostSeeds = nullptr;
    DeviceBuffer<RandomCheckpoint> deviceSeeds;

public:
    SeedBatch() = default;
    SeedBatch(const SeedBatch&) = delete;
    SeedBatch& operator=(const SeedBatch&) = delete;
    ~SeedBatch() {
        if (stream) cudaStreamSynchronize(stream);
        if (hostSeeds) cudaFreeHost(hostSeeds);
        if (stream) cudaStreamDestroy(stream);
    }

    void initialize(size_t count) {
        checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking), "Creating CUDA stream");
        checkCuda(cudaMallocHost(&hostSeeds, count * sizeof(RandomCheckpoint)),
                  "Allocating pinned random checkpoints");
        deviceSeeds.allocate(count);
    }

    void wait() const {
        checkCuda(cudaStreamSynchronize(stream), "Computing form factors");
    }

    void prepare(RandomSequence& sequence, const int* counts, size_t count) {
        wait();
        for (size_t k = 0; k < count; ++k) {
            sequence.reserve(hostSeeds[k], counts[k] * DRAWS_PER_PAIR);
        }
        checkCuda(cudaMemcpyAsync(deviceSeeds.get(), hostSeeds, count * sizeof(RandomCheckpoint),
                                  cudaMemcpyHostToDevice, stream), "Uploading random checkpoints");
    }

    cudaStream_t getStream() const { return stream; }
    const RandomCheckpoint* getSeeds() const { return deviceSeeds.get(); }
};

struct SimulationState {
    idx_t numTriangles;
    int numTimesteps;
    idx_t sourceIndex;
    val_t reflectivity;
    std::vector<val_t> distances;
    std::vector<uint8_t> energy;
    unsigned long long nonZeroKij = 0;

    DeviceBuffer<DeviceTriangle> triangles;
    DeviceBuffer<DeviceNode> nodes;
    DeviceBuffer<idx_t> indices;
    DeviceBuffer<val_t> weights;
    DeviceBuffer<uint8_t> delays;
    DeviceBuffer<val_t> radiosity;
};

void initializeSimulation(SimulationState& state, int subdivisions, int timesteps,
                          size_t sourceIdx, val_t reflectivity) {
    const IcosphereMesh mesh(subdivisions, 10.0f);
    state.numTriangles = static_cast<idx_t>(mesh.triangles.size());
    state.numTimesteps = timesteps;
    state.sourceIndex = sourceIdx % state.numTriangles;
    state.reflectivity = reflectivity;
    state.distances.resize(state.numTriangles);

    printf("Generated icosphere mesh with %u triangles\n", state.numTriangles);
    printf("Building octree...\n");
    Octree octree;
    octree.build(mesh.triangles);

    std::vector<DeviceTriangle> surfaces;
    surfaces.reserve(state.numTriangles);
    for (const Triangle& tri : mesh.triangles) {
        surfaces.push_back({tri.a, tri.b - tri.a, tri.c - tri.a, tri.normal(), tri.center(), tri.area()});
    }
    std::vector<DeviceNode> nodes;
    std::vector<idx_t> indices;
    flattenOctree(octree, nodes, indices);

    state.triangles.upload(surfaces);
    state.nodes.upload(nodes);
    state.indices.upload(indices);
    const size_t pairs = static_cast<size_t>(state.numTriangles) * state.numTriangles;
    const size_t history = static_cast<size_t>(timesteps) * state.numTriangles;
    size_t freeBytes, totalBytes;
    checkCuda(cudaMemGetInfo(&freeBytes, &totalBytes), "Querying CUDA memory");
    // Leave space for checkpoints and the CUDA runtime. Managed memory also
    // allows meshes whose pair matrices exceed device memory to run on CUDA.
    const bool managed = pairs * (sizeof(val_t) + sizeof(uint8_t)) + history * sizeof(val_t) >
                         freeBytes * 9 / 10;
    state.weights.allocate(pairs, managed);
    state.delays.allocate(pairs, managed);
    state.radiosity.allocate(history, managed);
}

void computeFormFactors(SimulationState& state, bool validate) {
    printf("Computing time delays (Tau)...\n");
    printf("Computing form factors (Kij)...\n");

    const idx_t n = state.numTriangles;
    const idx_t blocksPerRow = (n + PAIRS_PER_BLOCK - 1) / PAIRS_PER_BLOCK;
    const size_t blocks = static_cast<size_t>(blocksPerRow) * n;
    DeviceBuffer<int> deviceCounts;
    deviceCounts.allocate(blocks);

    countPairs<<<blocks, BLOCK_SIZE>>>(state.triangles.get(), n, blocksPerRow, deviceCounts.get());
    checkCuda(cudaGetLastError(), "Launching pair culling");
    std::vector<int> counts(blocks);
    deviceCounts.download(counts.data(), blocks);

    DeviceBuffer<unsigned long long> nonZero;
    if (validate) {
        nonZero.allocate(1);
        checkCuda(cudaMemset(nonZero.get(), 0, sizeof(unsigned long long)), "Clearing validation counter");
        checkCuda(cudaDeviceSynchronize(), "Initializing validation counter");
    }

    constexpr size_t BATCH_BLOCKS = 4096;
    SeedBatch batches[2];
    batches[0].initialize(std::min(blocks, BATCH_BLOCKS));
    if (blocks > BATCH_BLOCKS) batches[1].initialize(std::min(blocks - BATCH_BLOCKS, BATCH_BLOCKS));

    RandomSequence sequence(42);
    for (size_t first = 0; first < blocks; first += BATCH_BLOCKS) {
        const size_t count = std::min(BATCH_BLOCKS, blocks - first);
        auto& batch = batches[(first / BATCH_BLOCKS) % 2];
        batch.prepare(sequence, counts.data() + first, count);
        formFactors<<<count, BLOCK_SIZE, 0, batch.getStream()>>>(
            state.triangles.get(), state.nodes.get(), state.indices.get(), n, blocksPerRow, first,
            deviceCounts.get(), batch.getSeeds(), state.weights.get(), state.delays.get(), nonZero.get());
        checkCuda(cudaGetLastError(), "Launching form factor kernel");
    }

    batches[0].wait();
    if (blocks > BATCH_BLOCKS) batches[1].wait();
    if (validate) nonZero.download(&state.nonZeroKij, 1);

    printf("  Progress: %u/%u triangles\n", state.numTriangles, state.numTriangles);
}

void runSimulation(const SimulationState& state) {
    printf("Running wave propagation simulation...\n");
    // Launch in stream order because a timestep consumes earlier radiosity.
    for (int t = 0; t < state.numTimesteps; ++t) {
        propagate<<<state.numTriangles, BLOCK_SIZE>>>(
            state.weights.get(), state.delays.get(), state.radiosity.get(), state.numTriangles,
            state.sourceIndex, t, state.numTimesteps / 2, state.reflectivity);
        checkCuda(cudaGetLastError(), "Launching wave propagation");
        if ((t + 1) % 10 == 0 || t + 1 == state.numTimesteps) {
            printf("  Timestep %d/%d\n", t + 1, state.numTimesteps);
        }
    }
    checkCuda(cudaDeviceSynchronize(), "Running wave propagation");
}

void computeDistances(SimulationState& state, bool validate) {
    printf("Computing distances via cross-correlation...\n");
    DeviceBuffer<val_t> distances;
    DeviceBuffer<uint8_t> energy;
    distances.allocate(state.numTriangles);
    if (validate) {
        state.energy.resize(state.numTriangles);
        energy.allocate(state.numTriangles);
    }

    const dim3 threads(CORR_TRIANGLES, CORR_LAGS);
    const idx_t blocks = (state.numTriangles + CORR_TRIANGLES - 1) / CORR_TRIANGLES;
    correlate<<<blocks, threads>>>(state.radiosity.get(), distances.get(), energy.get(),
                                   state.numTriangles, state.sourceIndex, state.numTimesteps);
    checkCuda(cudaGetLastError(), "Launching cross-correlation");
    distances.download(state.distances.data(), state.distances.size());
    if (validate) energy.download(state.energy.data(), state.energy.size());
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
    printf("  Non-zero distances: %d/%zu\n", nonZeroCount, static_cast<size_t>(state.numTriangles));

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    const auto receivedEnergy = static_cast<size_t>(std::count(state.energy.begin(), state.energy.end(), 1));
    printf("  Triangles receiving energy: %zu/%u\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    const size_t pairs = static_cast<size_t>(state.numTriangles) * state.numTriangles;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n", state.nonZeroKij, pairs,
           100.0f * state.nonZeroKij / static_cast<val_t>(pairs));

    if (state.nonZeroKij == 0) {
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
        uint32_t bits;
        memcpy(&bits, &state.distances[i], sizeof(bits));
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

int main(int argc, char** argv) try {
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

    if (timesteps < 0) {
        fprintf(stderr, "Timesteps must be non-negative\n");
        return 1;
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
    initializeSimulation(state, subdivisions, timesteps,
                         static_cast<size_t>(sourceIdx), reflectivity);

    printf("\n");

    // Precomputation
    auto startPre = std::chrono::high_resolution_clock::now();

    computeFormFactors(state, validate);

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

    computeDistances(state, validate);

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
    size_t memTau = n * n * sizeof(uint8_t);
    size_t memRad = t * n * sizeof(val_t);
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
} catch (const std::exception& error) {
    fprintf(stderr, "roomsim: %s\n", error.what());
    return 1;
}
