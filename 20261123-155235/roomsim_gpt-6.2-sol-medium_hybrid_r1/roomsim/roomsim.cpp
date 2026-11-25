/**
 * Room Response Simulation Benchmark
 * 
 * This is a hybrid MPI, OpenMP, and CUDA implementation of room impulse response
 * simulation using radiosity-based wave propagation. It models how sound/light
 * waves propagate between surfaces in a room by computing:
 * 
 * 1. Form factors (Kij) between all pairs of triangles based on visibility and geometry
 * 2. Wave propagation using radiosity equations with time delays (Tau)
 * 3. Distance estimation via cross-correlation of radiosity values
 * 
 * The implementation generates a procedural icosphere mesh to simulate a room.
 */

#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <unordered_map>
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

    __host__ __device__ constexpr Vec3() : x(0), y(0), z(0) {}
    __host__ __device__ constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}

    __host__ __device__ Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    __host__ __device__ Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    __host__ __device__ Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    __host__ __device__ Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    __host__ __device__ Vec3 operator-() const { return {-x, -y, -z}; }

    __host__ __device__ val_t dot(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return fmaf(z, o.z, fmaf(x, o.x, y * o.y));
#else
        return x * o.x + y * o.y + z * o.z;
#endif
    }

    __host__ __device__ Vec3 cross(const Vec3& o) const {
#ifdef __CUDA_ARCH__
        return {fmaf(y, o.z, -z * o.y),
                fmaf(z, o.x, -x * o.z),
                fmaf(x, o.y, -y * o.x)};
#else
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
#endif
    }

    __host__ __device__ val_t squaredNorm() const { return dot(*this); }
    __host__ __device__ val_t norm() const { return sqrtf(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    Triangle() = default;
    Triangle(const Vec3& a, const Vec3& b, const Vec3& c) : a(a), b(b), c(c) {
        const Vec3 ab = b - a;
        const Vec3 ac = c - a;
        // The serial mesh loop contracts the normal's x and y components with the
        // second product. Pin those roundings across OpenMP work partitions.
        const Vec3 cross(fmaf(-ab.z, ac.y, ab.y * ac.z),
                         fmaf(-ab.x, ac.z, ab.z * ac.x),
                         fmaf(ab.x, ac.y, -ab.y * ac.x));
        const val_t length = sqrtf(fmaf(cross.z, cross.z,
                                        fmaf(cross.x, cross.x, cross.y * cross.y)));
        _normal = length > EPSILON ? cross / length : Vec3();
    }

    Vec3 center() const { return (a + b + c) / 3.0f; }
    Vec3 normal() const { return _normal; }

    val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }
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

        // Filter each child independently, preserving its triangle order.
        const Vec3 childHalfSize = halfExtent * 0.5f;
        std::vector<size_t> childIndices[8];
        #pragma omp taskloop shared(childIndices, indices) grainsize(1) if(indices.size() >= 1024)
        for (int i = 0; i < 8; ++i) {
            Vec3 childCenter = center;
            childCenter.x += (i & 1) ? childHalfSize.x : -childHalfSize.x;
            childCenter.y += (i & 2) ? childHalfSize.y : -childHalfSize.y;
            childCenter.z += (i & 4) ? childHalfSize.z : -childHalfSize.z;
            for (size_t idx : indices) {
                if (triangleBoxOverlap(childCenter, childHalfSize, (*allTriangles)[idx]))
                    childIndices[i].push_back(idx);
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

        // Each task owns one subtree and its parent waits for completion.
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
                Octree* child = children[i].get();
                #pragma omp task shared(childIndices) firstprivate(child, i, childMin, childMax) if(childIndices[i].size() >= 128)
                child->buildNode(childIndices[i], childMin, childMax);
            }
        }
        #pragma omp taskwait
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
            newFaces.reserve(4 * faces.size());
            std::unordered_map<uint64_t, idx_t> midpointCache;
            midpointCache.reserve(3 * faces.size() / 2);

            auto getMidpoint = [&](idx_t i1, idx_t i2) -> idx_t {
                const uint64_t key = (static_cast<uint64_t>(std::min(i1, i2)) << 32) |
                                     std::max(i1, i2);
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
        triangles.resize(faces.size());
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < faces.size(); ++i) {
            const auto& face = faces[i];
            // Reverse winding to make normals point inward.
            triangles[i] = Triangle(vertices[face[2]], vertices[face[1]], vertices[face[0]]);
        }
    }
};

// Each tile needs a small checkpoint of the original MT19937 stream. The
// untempered words x obey x[k+624] = x[k+397] ^ twist(x[k], x[k+1]).
constexpr int MT_WORDS = 624;
constexpr int MT_MIDDLE = 397;
constexpr int MT_STRIDE = MT_WORDS - MT_MIDDLE;
constexpr int MT_DEGREE = 19937;
constexpr int POLY_WORDS = MT_DEGREE / 64 + 1;
using RandomWords = std::array<uint32_t, MT_WORDS>;
using Polynomial = std::array<uint64_t, POLY_WORDS>;

__host__ __device__ uint32_t twist(uint32_t a, uint32_t b) {
    const uint32_t joined = (a & 0x80000000u) | (b & 0x7fffffffu);
    return (joined >> 1) ^ ((0u - (b & 1u)) & 0x9908b0dfu);
}

void advanceRandomBlock(RandomWords& words) {
    #pragma omp simd
    for (int i = 0; i < MT_STRIDE; ++i)
        words[i] = words[i + MT_MIDDLE] ^ twist(words[i], words[i + 1]);
    #pragma omp simd
    for (int i = MT_STRIDE; i < 2 * MT_STRIDE; ++i)
        words[i] = words[i - MT_STRIDE] ^ twist(words[i], words[i + 1]);
    #pragma omp simd
    for (int i = 2 * MT_STRIDE; i < MT_WORDS - 1; ++i)
        words[i] = words[i - MT_STRIDE] ^ twist(words[i], words[i + 1]);
    words.back() = words[MT_MIDDLE - 1] ^ twist(words.back(), words.front());
}

uint32_t stepRandomWord(RandomWords& words, int& cursor) {
    const int next = (cursor + 1) % MT_WORDS;
    const uint32_t word = words[cursor];
    words[cursor] = words[(cursor + MT_MIDDLE) % MT_WORDS] ^ twist(word, words[next]);
    cursor = next;
    return word;
}

bool coefficient(const uint64_t* poly, int degree) {
    return (poly[degree / 64] >> (degree % 64)) & 1u;
}

void setCoefficient(uint64_t* poly, int degree) {
    poly[degree / 64] |= uint64_t{1} << (degree % 64);
}

void xorShifted(uint64_t* dest, const uint64_t* src, int words, int shift) {
    dest += shift / 64;
    const int bits = shift % 64;
    if (bits == 0) {
        #pragma omp simd
        for (int i = 0; i < words; ++i) dest[i] ^= src[i];
        return;
    }
    dest[0] ^= src[0] << bits;
    #pragma omp simd
    for (int i = 1; i < words; ++i)
        dest[i] ^= (src[i] << bits) | (src[i - 1] >> (64 - bits));
    dest[words] ^= src[words - 1] >> (64 - bits);
}

// Berlekamp–Massey recovers the transition polynomial from a bit of the MT
// sequence. This lets each rank jump directly to its part of the stream.
Polynomial randomCharacteristic(RandomWords words) {
    Polynomial history{};
    std::array<uint64_t, POLY_WORDS + 1> connection{1}, previous{1};
    int cursor = 0;
    int degree = 0;
    int previousDegree = 0;
    int lastUpdate = -1;
    for (int i = 0; i < 2 * MT_DEGREE; ++i) {
        for (int w = POLY_WORDS - 1; w > 0; --w)
            history[w] = (history[w] << 1) | (history[w - 1] >> 63);
        history[0] = (history[0] << 1) | (stepRandomWord(words, cursor) & 1u);

        uint64_t discrepancy = 0;
        #pragma omp simd reduction(^:discrepancy)
        for (int w = 0; w <= degree / 64; ++w)
            discrepancy ^= connection[w] & history[w];
        if (__builtin_parityll(discrepancy) == 0) continue;

        const auto saved = connection;
        xorShifted(connection.data(), previous.data(), previousDegree / 64 + 1, i - lastUpdate);
        if (2 * degree <= i) {
            previous = saved;
            previousDegree = degree;
            degree = i + 1 - degree;
            lastUpdate = i;
        }
    }

    Polynomial characteristic{};
    for (int i = 0; i <= degree; ++i) {
        if (coefficient(connection.data(), i)) setCoefficient(characteristic.data(), degree - i);
    }
    return characteristic;
}

RandomWords jumpRandomStream(RandomWords words, uint64_t offset) {
    const auto characteristic = randomCharacteristic(words);
    Polynomial power{1};
    // Binary exponentiation of x, reduced modulo the characteristic polynomial.
    for (int bit = 63 - __builtin_clzll(offset); bit >= 0; --bit) {
        std::array<uint64_t, 2 * POLY_WORDS> square{};
        const int shift = (offset >> bit) & 1u;
        for (int i = 0; i < MT_DEGREE; ++i) {
            if (coefficient(power.data(), i)) setCoefficient(square.data(), 2 * i + shift);
        }
        for (int i = 2 * MT_DEGREE - 1; i >= MT_DEGREE; --i) {
            if (coefficient(square.data(), i))
                xorShifted(square.data(), characteristic.data(), POLY_WORDS, i - MT_DEGREE);
        }
        std::copy_n(square.begin(), POLY_WORDS, power.begin());
    }

    RandomWords result{};
    int cursor = 0;
    for (int i = 0; i < MT_DEGREE; ++i) {
        if (coefficient(power.data(), i)) {
            #pragma omp simd
            for (int j = 0; j < MT_WORDS - cursor; ++j) result[j] ^= words[cursor + j];
            #pragma omp simd
            for (int j = MT_WORDS - cursor; j < MT_WORDS; ++j)
                result[j] ^= words[j - (MT_WORDS - cursor)];
        }
        stepRandomWord(words, cursor);
    }
    return result;
}

struct RandomStream {
    RandomWords words;
    int offset = 0;

    explicit RandomStream(uint64_t start) {
        words[0] = 42;
        for (uint32_t i = 1; i < MT_WORDS; ++i)
            words[i] = 1812433253u * (words[i - 1] ^ (words[i - 1] >> 30)) + i;
        advanceRandomBlock(words);
        // Twisting whole blocks is cheaper for nearby starting positions.
        if (start >= 16'000'000) {
            words = jumpRandomStream(words, start);
        } else {
            advance(start);
        }
    }

    void advance(uint64_t count) {
        count += offset;
        for (uint64_t i = 0; i < count / MT_WORDS; ++i) advanceRandomBlock(words);
        offset = count % MT_WORDS;
    }
};

// Depth-first order and an escape index let a GPU ray traverse the original
// octree without a private stack.
struct OctreeNode {
    Vec3 center, halfExtent;
    int escape, begin, end;
};

void flattenOctree(const Octree& tree, std::vector<OctreeNode>& nodes,
                   std::vector<idx_t>& indices) {
    const size_t node = nodes.size();
    const int begin = static_cast<int>(indices.size());
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    nodes.push_back({tree.center, tree.halfExtent, 0, begin, static_cast<int>(indices.size())});
    for (const auto& child : tree.children) {
        if (child) flattenOctree(*child, nodes, indices);
    }
    nodes[node].escape = static_cast<int>(nodes.size());
}

struct RayTriangle {
    Vec3 a, ab, ac;
};

__device__ bool intersectsBox(const OctreeNode& node, const Vec3& midpoint,
                              const Vec3& d, const Vec3& ad) {
    const Vec3 c = midpoint - node.center;
    const Vec3 h = node.halfExtent;
    return fabsf(c.x) <= h.x + ad.x && fabsf(c.y) <= h.y + ad.y && fabsf(c.z) <= h.z + ad.z
        && fabsf(fmaf(d.y, c.z, -d.z * c.y)) <= fmaf(h.y, ad.z, h.z * ad.y) + EPSILON
        && fabsf(fmaf(d.z, c.x, -d.x * c.z)) <= fmaf(h.z, ad.x, h.x * ad.z) + EPSILON
        && fabsf(fmaf(d.x, c.y, -d.y * c.x)) <= fmaf(h.x, ad.y, h.y * ad.x) + EPSILON;
}

__device__ bool intersectsTriangle(const RayTriangle& tri, const Vec3& from,
                                   const Vec3& dir, val_t rayEnd) {
    const Vec3 p = dir.cross(tri.ac);
    const val_t det = tri.ab.dot(p);
    if (fabsf(det) < EPSILON) return false;
    const val_t invDet = 1.0f / det;
    const Vec3 fromA = from - tri.a;
    const val_t u = fromA.dot(p) * invDet;
    if (u < ZERO || u > ONE) return false;
    const Vec3 q = fromA.cross(tri.ab);
    const val_t v = dir.dot(q) * invDet;
    if (v < ZERO || u + v > ONE) return false;
    const val_t distance = tri.ac.dot(q) * invDet;
    return distance > EPSILON && distance < rayEnd;
}

__device__ bool isRayBlocked(const Vec3& from, const Vec3& diff, val_t rayLen,
                             const RayTriangle* triangles, const OctreeNode* nodes,
                             const idx_t* indices, idx_t i, idx_t j) {
    if (rayLen < EPSILON) return true;
    const Vec3 dir = diff / rayLen;
    const Vec3 d = diff * 0.5f;
    const Vec3 ad(fabsf(d.x), fabsf(d.y), fabsf(d.z));
    const Vec3 midpoint = from + d;
    const val_t rayEnd = rayLen - EPSILON;
    const int end = nodes[0].escape;
    for (int k = 0; k < end;) {
        const OctreeNode& node = nodes[k];
        if (k != 0 && !intersectsBox(node, midpoint, d, ad)) {
            k = node.escape;
            continue;
        }
        for (int p = node.begin; p < node.end; ++p) {
            const idx_t index = indices[p];
            if (index != i && index != j &&
                intersectsTriangle(triangles[index], from, dir, rayEnd)) return true;
        }
        ++k;
    }
    return false;
}

__device__ val_t uniformRandom(uint32_t word) {
    word ^= word >> 11;
    word ^= (word << 7) & 0x9d2c5680u;
    word ^= (word << 15) & 0xefc60000u;
    word ^= word >> 18;
    return fminf(__uint2float_rn(word) * 0x1p-32f, 0x1.fffffep-1f);
}

__device__ Vec3 randomPoint(const RayTriangle& tri, const uint32_t* words) {
    val_t u = uniformRandom(words[0]);
    val_t v = uniformRandom(words[1]);
    if (u + v > ONE) {
        u = ONE - u;
        v = ONE - v;
    }
    return {fmaf(tri.ac.x, v, fmaf(tri.ab.x, u, tri.a.x)),
            fmaf(tri.ac.y, v, fmaf(tri.ab.y, u, tri.a.y)),
            fmaf(tri.ac.z, v, fmaf(tri.ab.z, u, tri.a.z))};
}

__device__ val_t rayContribution(const RayTriangle* triangles, const Vec3* normals,
                                 const OctreeNode* nodes, const idx_t* indices,
                                 int i, int j, const uint32_t* words) {
    const Vec3 from = randomPoint(triangles[i], words);
    const Vec3 diff = randomPoint(triangles[j], words + 2) - from;
    const val_t distSqr = diff.squaredNorm();
    if (distSqr < EPSILON) return ZERO;
    const val_t distance = sqrtf(distSqr);
    const val_t cosineI = diff.dot(normals[i]) / distance;
    const val_t cosineJ = (-diff).dot(normals[j]) / distance;
    if (cosineI <= ZERO || cosineJ <= ZERO ||
        isRayBlocked(from, diff, distance, triangles, nodes, indices, i, j)) return ZERO;
    return (cosineI * cosineJ) / (PI * distSqr);
}

constexpr int TILE_PAIRS = 64;
constexpr int DRAWS_PER_PAIR = NUM_RAYS * 4;
constexpr int CUDA_THREADS = 256;
constexpr int BATCH_TILES = 4096;

struct FormFactorTile {
    uint64_t mask;
    int offset;
    uint32_t words[MT_WORDS];
};

__global__ void computeFormFactors(const FormFactorTile* tiles, size_t firstTile,
                                    int tilesPerRow, int firstRow, int n, int timesteps,
                                    const RayTriangle* triangles, const Vec3* normals,
                                    const Vec3* centers, const val_t* areas,
                                    const OctreeNode* nodes, const idx_t* indices,
                                    val_t* weights, uint8_t* delays,
                                    unsigned long long* nonzero, int* minDelay) {
    __shared__ uint32_t words[MT_WORDS - 1 + DRAWS_PER_PAIR * TILE_PAIRS];
    __shared__ val_t contributions[TILE_PAIRS * NUM_RAYS];
    __shared__ int counts[TILE_PAIRS], minimums[TILE_PAIRS];

    const auto& tile = tiles[blockIdx.x];
    const uint64_t mask = tile.mask;
    const int thread = threadIdx.x;
    if (mask != 0) {
        for (int k = thread; k < MT_WORDS; k += blockDim.x) words[k] = tile.words[k];
        __syncthreads();
        const int end = tile.offset + __popcll(mask) * DRAWS_PER_PAIR;
        for (int k = MT_WORDS; k < end; k += MT_STRIDE) {
            const int index = k + thread;
            if (thread < MT_STRIDE && index < end)
                words[index] = words[index - MT_STRIDE] ^
                               twist(words[index - MT_WORDS], words[index - MT_WORDS + 1]);
            __syncthreads();
        }
    }

    const size_t tileIdx = firstTile + blockIdx.x;
    const int row = static_cast<int>(tileIdx / tilesPerRow);
    const int i = firstRow + row;
    const int firstJ = static_cast<int>(tileIdx % tilesPerRow) * TILE_PAIRS;
    for (int sample = thread; sample < TILE_PAIRS * NUM_RAYS; sample += blockDim.x) {
        const int pair = sample / NUM_RAYS;
        const uint64_t bit = uint64_t{1} << pair;
        val_t value = ZERO;
        if (mask & bit) {
            const int draw = tile.offset + __popcll(mask & (bit - 1)) * DRAWS_PER_PAIR
                             + (sample % NUM_RAYS) * 4;
            value = rayContribution(triangles, normals, nodes, indices, i, firstJ + pair, words + draw);
        }
        contributions[sample] = value;
    }
    __syncthreads();

    if (thread < TILE_PAIRS) {
        const int j = firstJ + thread;
        int delay = timesteps;
        int count = 0;
        if (j < n) {
            val_t kij = ZERO;
            #pragma unroll
            for (int ray = 0; ray < NUM_RAYS; ++ray)
                kij += contributions[thread * NUM_RAYS + ray];
            kij *= INV_NUM_RAYS;
            count = kij > EPSILON;
            if (kij > ZERO) {
                delay = static_cast<int>(ceilf((centers[i] - centers[j]).norm() * INV_WAVE_SPEED));
            }
            const size_t pair = static_cast<size_t>(row) * n + j;
            weights[pair] = fminf(kij * areas[j], ONE);
            delays[pair] = kij > ZERO && delay < timesteps ? delay : 0;
        }
        counts[thread] = count;
        minimums[thread] = delay;
    }
    __syncthreads();

    if (thread < 32) {
        int count = counts[thread] + counts[thread + 32];
        int delay = min(minimums[thread], minimums[thread + 32]);
        for (int step = 16; step > 0; step /= 2) {
            count += __shfl_down_sync(0xffffffff, count, step);
            delay = min(delay, __shfl_down_sync(0xffffffff, delay, step));
        }
        if (thread == 0) {
            atomicAdd(nonzero, static_cast<unsigned long long>(count));
            atomicMin(minDelay, delay);
        }
    }
}

constexpr int PROPAGATION_THREADS = 128;
// A radius-10 room has a maximum geometric delay of 40 timesteps.
// Round the history capacity up to a power of two for cheap indexing.
constexpr int HISTORY_STEPS = 64;
static_assert(HISTORY_STEPS > 2 * ROOM_RADIUS * INV_WAVE_SPEED);

__global__ void propagate(const val_t* weights, const uint8_t* delays,
                          const val_t* history, val_t* output, int n, int begin,
                          int firstRow, int source, int timeOff, val_t reflectivity,
                          int firstArrival, int historyMask) {
    __shared__ val_t contributions[PROPAGATION_THREADS];
    const int row = blockIdx.x;
    const int t = begin + blockIdx.y;
    const int thread = threadIdx.x;
    val_t sum = ZERO;

    if (t >= firstArrival) {
        const size_t offset = static_cast<size_t>(row) * n;
        for (int base = 0; base < n; base += PROPAGATION_THREADS) {
            const int j = base + thread;
            val_t contribution = ZERO;
            if (j < n) {
                const int delay = delays[offset + j];
                if (delay != 0 && delay <= t) {
                    const int inputTime = (t - delay) & historyMask;
                    const val_t radiosity = history[static_cast<size_t>(inputTime) * n + j];
                    if (!(radiosity <= ZERO)) contribution = weights[offset + j] * radiosity;
                }
            }
            contributions[thread] = contribution;
            __syncthreads();
            if (thread == 0) {
                // Maintain the original emitter order for floating-point addition.
                #pragma unroll 8
                for (int k = 0; k < PROPAGATION_THREADS; ++k) sum += contributions[k];
            }
            __syncthreads();
        }
    }

    if (thread == 0) {
        const val_t emission = (firstRow + row == source && t < timeOff) ? ONE : ZERO;
        output[static_cast<size_t>(blockIdx.y) * gridDim.x + row] =
            fmaf(reflectivity, sum, emission);
    }
}

__global__ void unpackHistory(const val_t* gathered, val_t* recent, val_t* sourceHistory,
                              int n, int begin, int batch, int source,
                              const int* partitionStarts, const int* partitionSizes) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= n) return;

    const int first = partitionStarts[j];
    const int count = partitionSizes[j];
    for (int t = 0; t < batch; ++t) {
        const val_t value = gathered[static_cast<size_t>(first) * batch + t * count + j - first];
        recent[static_cast<size_t>((begin + t) & (HISTORY_STEPS - 1)) * n + j] = value;
        if (j == source) sourceHistory[begin + t] = value;
    }
}

__global__ void correlate(const val_t* history, const val_t* sourceHistory, val_t* distances,
                          int rows, int timesteps, int sourceStride, bool validate, int* received) {
    extern __shared__ val_t correlations[];
    const int i = blockIdx.x;

    // Lags are independent. Each thread accumulates its lag in time order.
    val_t best = ZERO;
    int bestLag = 0;
    for (int lag = threadIdx.x; lag < timesteps; lag += blockDim.x) {
        val_t sum = ZERO;
        for (int tt = lag; tt < timesteps; ++tt) {
            const val_t pB = history[static_cast<size_t>(tt) * rows + i];
            const val_t pS = sourceHistory[static_cast<size_t>(tt - lag) * sourceStride];
            sum += pS * pB;
        }

        if (sum > best) {
            best = sum;
            bestLag = lag;
        }
    }

    // Reduce maxima with the same first-lag tie break as the original loop.
    int* lags = reinterpret_cast<int*>(correlations + blockDim.x);
    correlations[threadIdx.x] = best;
    lags[threadIdx.x] = bestLag;
    __syncthreads();
    for (int step = blockDim.x / 2; step > 0; step /= 2) {
        if (threadIdx.x < step) {
            const int other = threadIdx.x + step;
            if (correlations[other] > correlations[threadIdx.x] ||
                (correlations[other] == correlations[threadIdx.x] && lags[other] < lags[threadIdx.x])) {
                correlations[threadIdx.x] = correlations[other];
                lags[threadIdx.x] = lags[other];
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) distances[blockIdx.x] = WAVE_SPEED * lags[0];

    if (validate) {
        bool energy = false;
        for (int tt = threadIdx.x; tt < timesteps; tt += blockDim.x) {
            if (history[static_cast<size_t>(tt) * rows + i] > EPSILON) {
                energy = true;
                break;
            }
        }
        if (__syncthreads_or(energy) && threadIdx.x == 0) atomicAdd(received, 1);
    }
}

void checkCuda(cudaError_t status, const char* expression) {
    if (status == cudaSuccess) return;
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    fprintf(stderr, "Rank %d: %s failed: %s\n", rank, expression, cudaGetErrorString(status));
    MPI_Abort(MPI_COMM_WORLD, 1);
    std::abort();
}

#define CUDA_CHECK(expression) checkCuda((expression), #expression)

class Stream {
public:
    cudaStream_t handle = nullptr;

    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&handle, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(handle); }
    Stream(const Stream&) = delete;
    Stream& operator=(const Stream&) = delete;

    void wait() const { CUDA_CHECK(cudaStreamSynchronize(handle)); }
};

template<typename T, bool Pinned = false>
class Buffer {
public:
    T* data = nullptr;

    Buffer() = default;
    explicit Buffer(size_t count) { allocate(count); }
    ~Buffer() {
        if (data == nullptr) return;
        if constexpr (Pinned) cudaFreeHost(data);
        else cudaFree(data);
    }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    void allocate(size_t count) {
        if (count == 0) return;
        if constexpr (Pinned) CUDA_CHECK(cudaMallocHost(&data, count * sizeof(T)));
        else CUDA_CHECK(cudaMalloc(&data, count * sizeof(T)));
    }

    void upload(const T* input, size_t count, const Stream& stream) const {
        if (count != 0)
            CUDA_CHECK(cudaMemcpyAsync(data, input, count * sizeof(T),
                                       cudaMemcpyHostToDevice, stream.handle));
    }

    void download(T* output, size_t count, const Stream& stream) const {
        if (count != 0)
            CUDA_CHECK(cudaMemcpyAsync(output, data, count * sizeof(T),
                                       cudaMemcpyDeviceToHost, stream.handle));
    }
};

void selectDevice() {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, 0, MPI_INFO_NULL, &local);
    int localRank;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);

    int devices;
    CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (devices == 0) checkCuda(cudaErrorNoDevice, "GPU selection");
    // Also works when the scheduler exposes only the GPU assigned to this rank.
    CUDA_CHECK(cudaSetDevice(localRank % devices));
}

struct Partition {
    std::vector<int> sizes, starts;
    int first, rows;

    Partition(int n, int rank, int ranks) : sizes(ranks), starts(ranks) {
        for (int r = 0; r < ranks; ++r) {
            sizes[r] = n / ranks + (r < n % ranks);
            starts[r] = r * (n / ranks) + std::min(r, n % ranks);
        }
        first = starts[rank];
        rows = sizes[rank];
    }
};

struct SimulationState {
    const int numTriangles;
    const int numTimesteps;
    const int sourceIndex;
    const val_t reflectivity;
    const Partition partition;

    Stream stream;
    Buffer<val_t> weights, history, sourceHistory;
    Buffer<uint8_t> delays;
    int minDelay = 0;
    uint64_t nonZeroKij = 0;
    int receivedEnergy = 0;
    std::vector<val_t> distances;

    SimulationState(int n, int timesteps, size_t source, val_t reflectivity, int rank, int ranks)
        : numTriangles(n), numTimesteps(timesteps), sourceIndex(source % n),
          reflectivity(reflectivity), partition(n, rank, ranks),
          weights(static_cast<size_t>(partition.rows) * n),
          history(static_cast<size_t>(timesteps) * partition.rows),
          sourceHistory(ranks > 1 && partition.rows > 0 ? timesteps : 0),
          delays(static_cast<size_t>(partition.rows) * n) {}
};

void precompute(SimulationState& state, const std::vector<Triangle>& triangles,
                const std::vector<OctreeNode>& nodes, const std::vector<idx_t>& indices) {
    const int n = state.numTriangles;
    const int rows = state.partition.rows;
    const int firstRow = state.partition.first;
    const int tilesPerRow = (n + TILE_PAIRS - 1) / TILE_PAIRS;
    const size_t numTiles = static_cast<size_t>(rows) * tilesPerRow;
    std::vector<RayTriangle> rayTriangles(n);
    std::vector<Vec3> normals(n), centers(n);
    std::vector<val_t> areas(n);
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < n; ++i) {
        const auto& tri = triangles[i];
        rayTriangles[i] = {tri.a, tri.b - tri.a, tri.c - tri.a};
        normals[i] = tri.normal();
        centers[i] = tri.center();
        areas[i] = tri.area();
    }

    // Culled pairs did not draw random numbers in the serial traversal.
    // Compute each tile's mask using the host geometry, then prefix the
    // draw counts in the original global receiver/emitter order.
    std::vector<uint64_t> masks(numTiles);
    uint64_t localDraws = 0;
    #pragma omp parallel for schedule(static) reduction(+:localDraws)
    for (int row = 0; row < rows; ++row) {
        const int i = firstRow + row;
        const Vec3 normal = normals[i];
        for (int tile = 0; tile < tilesPerRow; ++tile) {
            const int firstJ = tile * TILE_PAIRS;
            const int count = std::min(TILE_PAIRS, n - firstJ);
            uint64_t mask = 0;
            for (int p = 0; p < count; ++p) {
                const int j = firstJ + p;
                if (i != j && normal.dot(normals[j]) <= 0.99f) mask |= uint64_t{1} << p;
            }
            masks[static_cast<size_t>(row) * tilesPerRow + tile] = mask;
            localDraws += DRAWS_PER_PAIR * __builtin_popcountll(mask);
        }
    }

    uint64_t firstDraw = 0;
    if (state.partition.sizes.size() > 1) {
        MPI_Exscan(&localDraws, &firstDraw, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
        if (firstRow == 0) firstDraw = 0;
    }
    int localMin = state.numTimesteps;
    unsigned long long localNonZero = 0;

    if (numTiles != 0) {
        Buffer<RayTriangle> dTriangles(n);
        Buffer<Vec3> dNormals(n), dCenters(n);
        Buffer<val_t> dAreas(n);
        Buffer<OctreeNode> dNodes(nodes.size());
        Buffer<idx_t> dIndices(indices.size());
        Buffer<unsigned long long> dNonZero(1);
        Buffer<int> dMin(1);
        const auto& setup = state.stream;
        dTriangles.upload(rayTriangles.data(), n, setup);
        dNormals.upload(normals.data(), n, setup);
        dCenters.upload(centers.data(), n, setup);
        dAreas.upload(areas.data(), n, setup);
        dNodes.upload(nodes.data(), nodes.size(), setup);
        dIndices.upload(indices.data(), indices.size(), setup);
        dMin.upload(&localMin, 1, setup);
        CUDA_CHECK(cudaMemsetAsync(dNonZero.data, 0, sizeof(localNonZero), setup.handle));
        setup.wait();

        const int capacity = static_cast<int>(std::min<size_t>(BATCH_TILES, numTiles));
        Stream streams[2];
        Buffer<FormFactorTile, true> host[2];
        Buffer<FormFactorTile> device[2];
        for (int i = 0; i < (numTiles > BATCH_TILES ? 2 : 1); ++i) {
            host[i].allocate(capacity);
            device[i].allocate(capacity);
        }

        RandomStream rng(firstDraw);
        int slot = 0;
        for (size_t base = 0; base < numTiles; base += capacity, slot ^= 1) {
            const auto& stream = streams[slot];
            stream.wait();
            const int count = static_cast<int>(std::min<size_t>(capacity, numTiles - base));
            for (int k = 0; k < count; ++k) {
                auto& tile = host[slot].data[k];
                tile.mask = masks[base + k];
                if (tile.mask == 0) continue;
                tile.offset = rng.offset;
                std::copy(rng.words.begin(), rng.words.end(), tile.words);
                rng.advance(DRAWS_PER_PAIR * __builtin_popcountll(tile.mask));
            }
            device[slot].upload(host[slot].data, count, stream);
            computeFormFactors<<<count, CUDA_THREADS, 0, stream.handle>>>(
                device[slot].data, base, tilesPerRow, firstRow, n, state.numTimesteps,
                dTriangles.data, dNormals.data, dCenters.data, dAreas.data, dNodes.data, dIndices.data,
                state.weights.data, state.delays.data, dNonZero.data, dMin.data);
            CUDA_CHECK(cudaGetLastError());
        }
        for (const auto& stream : streams) stream.wait();
        dMin.download(&localMin, 1, setup);
        dNonZero.download(&localNonZero, 1, setup);
        setup.wait();
    }

    MPI_Allreduce(&localMin, &state.minDelay, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
    const uint64_t count = localNonZero;
    MPI_Reduce(&count, &state.nonZeroKij, 1, MPI_UINT64_T, MPI_SUM, 0, MPI_COMM_WORLD);
}

void runSimulation(SimulationState& state) {
    const int n = state.numTriangles;
    const int timesteps = state.numTimesteps;
    const auto& part = state.partition;
    const auto& stream = state.stream;
    const bool distributed = part.sizes.size() > 1;
    const int batchSize = std::clamp(state.minDelay, 1, 8);
    Buffer<val_t> recent(distributed && part.rows > 0 ? static_cast<size_t>(HISTORY_STEPS) * n : 0);
    const val_t* inputs = distributed ? recent.data : state.history.data;
    const int historyMask = distributed ? HISTORY_STEPS - 1 : -1;
    Buffer<val_t, true> send(distributed ? batchSize * part.rows : 0);
    Buffer<val_t, true> recv(distributed ? batchSize * n : 0);
    Buffer<val_t> gathered(distributed && part.rows > 0 ? batchSize * n : 0);
    Buffer<int> dStarts(distributed && part.rows > 0 ? n : 0);
    Buffer<int> dSizes(distributed && part.rows > 0 ? n : 0);
    std::vector<int> counts(part.sizes.size()), displacements(part.sizes.size());
    if (distributed && part.rows > 0) {
        std::vector<int> starts(n), sizes(n);
        for (size_t r = 0; r < part.sizes.size(); ++r) {
            std::fill_n(starts.begin() + part.starts[r], part.sizes[r], part.starts[r]);
            std::fill_n(sizes.begin() + part.starts[r], part.sizes[r], part.sizes[r]);
        }
        dStarts.upload(starts.data(), n, stream);
        dSizes.upload(sizes.data(), n, stream);
        stream.wait();
    }

    // All dependencies cross a batch boundary because every contributing
    // delay is at least minDelay. Exchange just the new samples each batch.
    // Keep only recent remote rows for propagation and the full source
    // signal and local rows for correlation.
    for (int begin = 0; begin < timesteps;) {
        const int batch = std::min(batchSize, timesteps - begin);
        val_t* output = state.history.data;
        if (part.rows > 0) {
            output += static_cast<size_t>(begin) * part.rows;
            const dim3 grid(part.rows, batch);
            propagate<<<grid, PROPAGATION_THREADS, 0, stream.handle>>>(
                state.weights.data, state.delays.data, inputs, output, n, begin, part.first,
                state.sourceIndex, timesteps / 2, state.reflectivity, state.minDelay, historyMask);
            CUDA_CHECK(cudaGetLastError());
        }

        if (distributed) {
            if (part.rows > 0)
                CUDA_CHECK(cudaMemcpyAsync(send.data, output, batch * part.rows * sizeof(val_t),
                                           cudaMemcpyDeviceToHost, stream.handle));
            stream.wait();
            for (size_t r = 0; r < part.sizes.size(); ++r) {
                counts[r] = batch * part.sizes[r];
                displacements[r] = batch * part.starts[r];
            }
            MPI_Allgatherv(send.data, batch * part.rows, MPI_FLOAT, recv.data,
                           counts.data(), displacements.data(), MPI_FLOAT, MPI_COMM_WORLD);
            if (part.rows > 0) {
                gathered.upload(recv.data, batch * n, stream);
                unpackHistory<<<(n + CUDA_THREADS - 1) / CUDA_THREADS, CUDA_THREADS, 0, stream.handle>>>(
                    gathered.data, recent.data, state.sourceHistory.data,
                    n, begin, batch, state.sourceIndex, dStarts.data, dSizes.data);
                CUDA_CHECK(cudaGetLastError());
            }
        }
        begin += batch;
    }
    stream.wait();
}

void computeDistances(SimulationState& state, bool validate, int rank) {
    const auto& part = state.partition;
    const auto& stream = state.stream;
    std::vector<val_t> local(part.rows);
    int receivedHere = 0;
    if (part.rows > 0) {
        const bool distributed = part.sizes.size() > 1;
        const val_t* source = state.sourceHistory.data;
        if (!distributed && state.numTimesteps > 0) source = state.history.data + state.sourceIndex;
        const int sourceStride = distributed ? 1 : part.rows;
        constexpr int threads = 64;
        Buffer<val_t> dDistances(part.rows);
        Buffer<int> received(1);
        CUDA_CHECK(cudaMemsetAsync(received.data, 0, sizeof(int), stream.handle));
        correlate<<<part.rows, threads, threads * (sizeof(val_t) + sizeof(int)), stream.handle>>>(
            state.history.data, source, dDistances.data, part.rows, state.numTimesteps,
            sourceStride, validate, received.data);
        CUDA_CHECK(cudaGetLastError());
        dDistances.download(local.data(), local.size(), stream);
        if (validate) received.download(&receivedHere, 1, stream);
        stream.wait();
    }

    if (rank == 0) state.distances.resize(state.numTriangles);
    MPI_Gatherv(local.data(), part.rows, MPI_FLOAT, state.distances.data(),
                part.sizes.data(), part.starts.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
    if (validate) MPI_Reduce(&receivedHere, &state.receivedEnergy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
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

    for (int i = 0; i < state.numTriangles; ++i) {
        val_t d = state.distances[i];
        if (d < 0) {
            allNonNegative = false;
            printf("  ERROR: Negative distance at triangle %d: %f\n", i, d);
        }
        if (!std::isfinite(d)) {
            printf("  ERROR: Non-finite distance at triangle %d: %f\n", i, d);
            return false;
        }
        minDist = std::min(minDist, d);
        maxDist = std::max(maxDist, d);
        sumDist += d;
        if (d > EPSILON) nonZeroCount++;
    }

    printf("  Distance range: [%.4f, %.4f]\n", minDist, maxDist);
    printf("  Average distance: %.4f\n", sumDist / static_cast<val_t>(state.numTriangles));
    printf("  Non-zero distances: %d/%d\n", nonZeroCount, state.numTriangles);

    // Check source distance is zero or very small
    val_t srcDist = state.distances[state.sourceIndex];
    if (srcDist > WAVE_SPEED * 2) {
        printf("  WARNING: Source triangle distance is non-zero: %.4f\n", srcDist);
    }

    printf("  Triangles receiving energy: %d/%d\n", state.receivedEnergy, state.numTriangles);
    if (state.receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    const size_t numPairs = static_cast<size_t>(state.numTriangles) * state.numTriangles;
    const uint64_t nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %" PRIu64 "/%zu (%.2f%%)\n",
           nonZeroKij, numPairs, 100.0f * nonZeroKij / static_cast<val_t>(numPairs));

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
    for (int i = 0; i < state.numTriangles; ++i) {
        uint32_t bits;
        std::memcpy(&bits, &state.distances[i], sizeof(bits));
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
        if (root) fprintf(stderr, "Timesteps must be non-negative.\n");
        return 1;
    }

    selectDevice();
    if (std::getenv("OMP_NUM_THREADS") == nullptr)
        omp_set_num_threads(std::min(8, omp_get_max_threads()));

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

    IcosphereMesh mesh(subdivisions, ROOM_RADIUS);
    const int nTriangles = static_cast<int>(mesh.triangles.size());
    if (root) {
        printf("Generated icosphere mesh with %d triangles\n", nTriangles);
        printf("Building octree...\n\n");
    }

    std::vector<OctreeNode> nodes;
    std::vector<idx_t> indices;
    {
        Octree tree;
        #pragma omp parallel
        {
            #pragma omp single
            tree.build(mesh.triangles);
        }
        flattenOctree(tree, nodes, indices);
    }

    SimulationState state(nTriangles, timesteps, static_cast<size_t>(sourceIdx), reflectivity, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);

    const auto startPre = std::chrono::steady_clock::now();
    if (root) printf("Computing time delays (Tau)...\nComputing form factors (Kij)...\n");
    precompute(state, mesh.triangles, nodes, indices);
    const auto endPre = std::chrono::steady_clock::now();
    const auto preDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endPre - startPre).count();
    if (root) printf("Precomputation time: %ld ms\n\n", preDuration);

    const auto startSim = std::chrono::steady_clock::now();
    if (root) printf("Running wave propagation simulation...\n");
    runSimulation(state);
    MPI_Barrier(MPI_COMM_WORLD);
    const auto endSim = std::chrono::steady_clock::now();
    const auto simDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endSim - startSim).count();
    if (root) printf("Simulation time: %ld ms\n\n", simDuration);

    const auto startDist = std::chrono::steady_clock::now();
    if (root) printf("Computing distances via cross-correlation...\n");
    computeDistances(state, validate, rank);
    const auto endDist = std::chrono::steady_clock::now();
    const auto distDuration = std::chrono::duration_cast<std::chrono::milliseconds>(endDist - startDist).count();

    int result = 0;
    if (root) {
        printf("Distance computation time: %ld ms\n\n", distDuration);
        // Total time
        long totalTime = preDuration + simDuration + distDuration;
        printf("Total computation time: %ld ms\n", totalTime);

        // Performance metrics
        size_t n = state.numTriangles;
        size_t t = state.numTimesteps;
        double kijOps = static_cast<double>(n * n);
        double simOps = kijOps * t;
        double distOps = static_cast<double>(n) * t * t;

        printf("\nPerformance:\n");
        printf("  Triangles: %zu\n", n);
        printf("  Timesteps: %zu\n", t);
        printf("  Form factor computations: %.2e\n", kijOps);
        printf("  Simulation operations: %.2e\n", simOps);
        printf("  Distance computations: %.2e\n", distOps);
        printf("  Total time per triangle: %.4f ms\n", static_cast<double>(totalTime) / n);

        // Combined storage for coefficients, delays and radiosity histories.
        size_t memKij = n * n * sizeof(val_t);
        size_t memTau = n * n * sizeof(uint8_t);
        size_t memRad = t * n * sizeof(val_t);
        if (ranks > 1) {
            const size_t activeRanks = std::min(ranks, nTriangles);
            memRad += activeRanks * (t + HISTORY_STEPS * n) * sizeof(val_t);
        }
        size_t totalMem = memKij + memTau + memRad;
        printf("  Memory usage: %.2f MB\n", totalMem / (1024.0 * 1024.0));

        // Hash
        uint64_t hash = computeHash(state);
        printf("  Result hash: %016" PRIX64 "\n", hash);
        printf("\n");

        // Print results for external validation
        if (printResults) {
            // Convert distances to double for output
            std::vector<double> distData(state.distances.begin(), state.distances.end());
            print_results(distData, "Distances");
        }

        if (validate && !validateResults(state)) result = 1;
    }
    if (ranks > 1) MPI_Bcast(&result, 1, MPI_INT, 0, MPI_COMM_WORLD);
    return result;
}

int main(int argc, char** argv) {
    int provided;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    int rank, ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) {
        if (rank == 0) fprintf(stderr, "MPI_THREAD_FUNNELED support is required.\n");
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    int result = 1;
    try {
        result = benchmark(argc, argv, rank, ranks);
    } catch (const std::exception& error) {
        fprintf(stderr, "Rank %d: %s\n", rank, error.what());
        MPI_Abort(MPI_COMM_WORLD, 1);
    }
    MPI_Finalize();
    return result;
}
