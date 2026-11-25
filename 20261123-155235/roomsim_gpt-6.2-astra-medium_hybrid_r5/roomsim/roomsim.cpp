/**
 * Room Response Simulation Benchmark
 *
 * This benchmark distributes receiver triangles across MPI ranks for room impulse response
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
#include <bitset>
#include <cuda_runtime.h>
#include <mpi.h>
#include <omp.h>
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

    __host__ __device__ val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    __host__ __device__ Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    __host__ __device__ val_t squaredNorm() const { return x * x + y * y + z * z; }
    __host__ __device__ val_t norm() const { return std::sqrt(squaredNorm()); }
    __host__ __device__ Vec3 normalized() const {
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

    __host__ __device__ Vec3 center() const { return (a + b + c) / 3.0f; }
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

// Depth-first order lets a ray skip an entire missed subtree in one step.
struct FlatNode {
    Vec3 center, halfExtent;
    int end, first, count;
    __device__ bool rayIntersectsBox(const Vec3& p1, const Vec3& p2) const {
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

void flatten(const Octree& tree, std::vector<FlatNode>& nodes, std::vector<idx_t>& indices) {
    size_t here = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, 0, static_cast<int>(indices.size()),
                     static_cast<int>(tree.triangleIndices.size())});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flatten(*child, nodes, indices);
    nodes[here].end = static_cast<int>(nodes.size());
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

__device__ Vec3 randomPointInTriangle(const Triangle& t, float u, float v) {
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

__device__ val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
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
// Distributed simulation and CUDA kernels
// ============================================================================

void checkCuda(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "roomsim: CUDA error: %s\n", cudaGetErrorString(err));
        if (!omp_in_parallel()) MPI_Abort(MPI_COMM_WORLD, 2);
        std::abort();
    }
}

template<class T>
class DeviceBuffer {
public:
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void allocate(size_t count) {
        if (count) checkCuda(cudaMalloc(&data, count * sizeof(T)));
    }
    void upload(const std::vector<T>& host) {
        allocate(host.size());
        if (!host.empty())
            checkCuda(cudaMemcpy(data, host.data(), host.size() * sizeof(T), cudaMemcpyHostToDevice));
    }
};

// A circular window of untempered MT19937 outputs. A jump applies x^offset
// modulo the transition's characteristic polynomial to all words in the window.
class SampleStream {
    static constexpr int words = 624, degree = 19937;
    std::array<uint32_t, words> window;
    int head = 0;

    uint32_t advance() {
        int next = (head + 1) % words;
        uint32_t old = window[head];
        uint32_t joined = (old & 0x80000000u) | (window[next] & 0x7fffffffu);
        window[head] = window[(head + 397) % words] ^ (joined >> 1) ^
                       ((joined & 1) ? 0x9908b0dfu : 0u);
        head = next;
        return old;
    }

    // With an aligned window, consecutive twists can be vectorized in two
    // runs: the second run reads words already updated by the first.
    void refill() {
        auto twist = [](uint32_t a, uint32_t b, uint32_t c) {
            uint32_t y = (a & 0x80000000u) | (b & 0x7fffffffu);
            return c ^ (y >> 1) ^ ((b & 1) ? 0x9908b0dfu : 0u);
        };
        for (int i = 0; i < words - 397; ++i)
            window[i] = twist(window[i], window[i+1], window[i+397]);
        for (int i = words - 397; i < words - 1; ++i)
            window[i] = twist(window[i], window[i+1], window[i+397-words]);
        window.back() = twist(window.back(), window.front(), window[396]);
        head = 0;
    }

    void jump(uint64_t offset) {
        // Exponents of the nonzero terms below the leading x^19937.
        static constexpr int terms[] = {
            0, 1189, 1416, 1585, 1643, 1870, 2493, 2773, 3000, 3227, 3454, 3681, 3908, 4135, 4362,
            4753, 5661, 6337, 6569, 7129, 7477, 7525, 7583, 7752, 7979, 8206, 9505, 9901, 9969,
            10128, 10693, 10761, 10920, 11089, 11147, 11157, 11215, 11321, 11374, 11384, 11485,
            11611, 11712, 11717, 11838, 11881, 11944, 11997, 12277, 12335, 12393, 12504, 12509,
            12620, 12673, 12731, 12736, 12789, 12905, 12958, 12963, 13137, 13185, 13190, 13243,
            13301, 13412, 13528, 13533, 13639, 13697, 13760, 13813, 13866, 14093, 14151, 14209,
            14320, 14325, 14436, 14547, 14552, 14605, 14721, 14774, 14779, 14953, 15001, 15006,
            15059, 15117, 15228, 15344, 15349, 15455, 15513, 15576, 15629, 15682, 15909, 15967,
            16025, 16136, 16141, 16252, 16363, 16368, 16421, 16537, 16590, 16595, 16817, 16822,
            16875, 16933, 17044, 17160, 17271, 17329, 17445, 17498, 17725, 17783, 17841, 17952,
            18068, 18179, 18237, 18406, 18633, 18691, 18860, 19087, 19314
        };
        std::bitset<2 * degree> poly;
        poly.set(0);
        int highest = 0;
        for (uint64_t bit = uint64_t(1) << 63; bit; bit >>= 1) {
            int shift = (offset & bit) != 0;
            for (int i = highest; i >= 0; --i) {
                bool value = poly[i];
                poly.reset(i);
                poly.set(2 * i + shift, value);
            }
            highest = 2 * highest + shift;
            for (int i = highest; i >= degree; --i) {
                if (!poly[i]) continue;
                poly.reset(i);
                for (int term : terms) poly.flip(i - degree + term);
            }
            highest = std::min(highest, degree - 1);
        }
        std::array<uint32_t, words> result{};
        for (int i = 0; i <= highest; ++i) {
            if (poly[i]) {
                int split = words - head;
                for (int j = 0; j < split; ++j) result[j] ^= window[head + j];
                for (int j = split; j < words; ++j) result[j] ^= window[j - split];
            }
            advance();
        }
        window = result;
        head = 0;
    }
public:
    explicit SampleStream(uint64_t offset) {
        window[0] = 42;
        for (int i = 1; i < words; ++i)
            window[i] = 1812433253u * (window[i-1] ^ (window[i-1] >> 30)) + i;
        for (int i = 0; i < words; ++i) advance();
        if (offset < 2000000) {
            for (uint64_t i = 0; i < offset; ++i) advance();
        } else {
            jump(offset);
        }
        std::rotate(window.begin(), window.begin() + head, window.end());
        head = 0;
    }

    float next() {
        if (head == words) refill();
        uint32_t y = window[head++];
        y ^= y >> 11;
        y ^= (y << 7) & 0x9d2c5680u;
        y ^= (y << 15) & 0xefc60000u;
        y ^= y >> 18;
        return std::min(static_cast<float>(y) * 0x1p-32f, 0x1.fffffep-1f);
    }
};

__device__ float traceSample(const Triangle* triangles, const FlatNode* nodes,
                             const idx_t* indices, int i, int j, float4 sample) {
    const Triangle& receiver = triangles[i];
    const Triangle& emitter = triangles[j];
    Vec3 from = randomPointInTriangle(receiver, sample.x, sample.y);
    Vec3 to = randomPointInTriangle(emitter, sample.z, sample.w);
    Vec3 ray = to - from;
    float squared = ray.squaredNorm();
    if (squared < EPSILON) return ZERO;
    float length = sqrtf(squared);
    float cosI = ray.dot(receiver._normal) / length;
    float cosJ = (-ray).dot(emitter._normal) / length;
    if (cosI <= ZERO || cosJ <= ZERO) return ZERO;

    Vec3 dir = ray / length;
    for (int k = 0; k < nodes[0].end;) {
        const FlatNode& node = nodes[k];
        if (k && !node.rayIntersectsBox(from, to)) {
            k = node.end;
            continue;
        }
        for (int q = 0; q < node.count; ++q) {
            int other = indices[node.first + q];
            if (other == i || other == j) continue;
            const Triangle& tri = triangles[other];
            float hit = rayTriangleIntersect(from, dir, tri.a, tri.b, tri.c);
            if (hit > EPSILON && hit < length - EPSILON) return ZERO;
        }
        ++k;
    }
    return (cosI * cosJ) / (PI * squared);
}

constexpr int blockSize = 128;

__global__ void formFactors(const Triangle* triangles, const FlatNode* nodes,
                            const idx_t* indices, const float* areas, const float4* samples,
                            int n, int firstRow, size_t start, int pairs, float* weights,
                            unsigned char* delays, unsigned long long* nonzero) {
    __shared__ float contribution[blockSize];
    int thread = threadIdx.x;
    int ray = thread % NUM_RAYS;
    int pair = (blockIdx.x * blockSize + thread) / NUM_RAYS;
    size_t entry = start + pair;
    int i = firstRow + entry / n, j = entry % n;
    float value = ZERO;
    if (pair < pairs) {
        float4 sample = samples[pair * NUM_RAYS + ray];
        if (sample.x >= ZERO) value = traceSample(triangles, nodes, indices, i, j, sample);
    }
    contribution[thread] = value;
    __syncthreads();
    bool positive = false;
    if (pair < pairs && ray == 0) {
        float kij = ZERO;
        for (int r = 0; r < NUM_RAYS; ++r) kij += contribution[thread + r];
        kij *= INV_NUM_RAYS;
        weights[entry] = fminf(kij * areas[j], ONE);
        // The mesh has radius 10, so all delays fit in one byte.
        delays[entry] = static_cast<unsigned char>(ceilf(
            (triangles[i].center() - triangles[j].center()).norm() * INV_WAVE_SPEED));
        positive = kij > EPSILON;
    }
    if (nonzero) {
        int count = __syncthreads_count(positive);
        if (!thread && count) atomicAdd(nonzero, static_cast<unsigned long long>(count));
    }
}

struct SimulationState {
    size_t numTriangles, numTimesteps, sourceIndex;
    int rank, ranks, first, rows, device, receivedEnergy = 0;
    unsigned long long nonZeroKij = 0;
    std::vector<int> counts, offsets;
    std::vector<float> distances;
    std::vector<Triangle> triangles;
    DeviceBuffer<Triangle> geometry;
    DeviceBuffer<float> areas, weights, recent, history;
    DeviceBuffer<unsigned char> delays;
    DeviceBuffer<FlatNode> nodes;
    DeviceBuffer<idx_t> indices;
};

// All triangle centers lie inside the sphere of radius 10. A new slice
// must not overwrite any of the preceding ceil(20 / WAVE_SPEED) slices.
constexpr int historySlots = 1 + static_cast<int>(20 * INV_WAVE_SPEED);

void initialize(SimulationState& s, int subdivisions, int steps, size_t source, int rank, int ranks) {
    MPI_Comm local;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local);
    int localRank, devices;
    MPI_Comm_rank(local, &localRank);
    MPI_Comm_free(&local);
    checkCuda(cudaGetDeviceCount(&devices));
    if (!devices) {
        fprintf(stderr, "roomsim requires a visible CUDA device\n");
        MPI_Abort(MPI_COMM_WORLD, 2);
    }
    s.device = localRank % devices;
    checkCuda(cudaSetDevice(s.device));
    s.triangles = IcosphereMesh(subdivisions, 10.0f).triangles;
    int n = s.triangles.size();
    s.numTriangles = n;
    s.numTimesteps = steps;
    s.sourceIndex = source % n;
    s.rank = rank;
    s.ranks = ranks;
    for (int r = 0; r < ranks; ++r) {
        s.counts.push_back(n / ranks + (r < n % ranks));
        s.offsets.push_back(r * (n / ranks) + std::min(r, n % ranks));
    }
    s.rows = s.counts[rank];
    s.first = s.offsets[rank];
    if (!rank) printf("Generated icosphere mesh with %d triangles\nBuilding octree...\n", n);
    Octree tree;
    tree.build(s.triangles);
    std::vector<FlatNode> nodes;
    std::vector<idx_t> indices;
    flatten(tree, nodes, indices);
    s.nodes.upload(nodes);
    s.indices.upload(indices);
    s.geometry.upload(s.triangles);
    std::vector<float> areas(n);
    int areaWorkers = std::min(omp_get_max_threads(), (n + 255) / 256);
    #pragma omp parallel for schedule(static) num_threads(areaWorkers)
    for (int j = 0; j < n; ++j) areas[j] = s.triangles[j].area();
    s.areas.upload(areas);
    s.weights.allocate(size_t(s.rows) * n);
    s.delays.allocate(size_t(s.rows) * n);
    s.recent.allocate(size_t(std::min(steps, historySlots)) * n);
    s.history.allocate(size_t(steps) * s.rows);
}

void precompute(SimulationState& s, bool validate) {
    const int n = s.numTriangles;
    auto active = [&](int i, int j) {
        return i != j && s.triangles[i]._normal.dot(s.triangles[j]._normal) <= 0.99f;
    };
    constexpr int batchSize = 4096;
    int workers = std::max(1, std::min({omp_get_max_threads(), s.rows,
        static_cast<int>((size_t(s.rows) * n + batchSize - 1) / batchSize)}));
    std::vector<uint64_t> offsets(s.rows + 1);
    #pragma omp parallel for schedule(static) num_threads(workers)
    for (int row = 0; row < s.rows; ++row) {
        uint64_t count = 0;
        for (int j = 0; j < n; ++j) count += active(s.first + row, j);
        offsets[row + 1] = count * (4 * NUM_RAYS);
    }
    for (int row = 0; row < s.rows; ++row) offsets[row + 1] += offsets[row];
    uint64_t base = 0;
    MPI_Exscan(&offsets.back(), &base, 1, MPI_UINT64_T, MPI_SUM, MPI_COMM_WORLD);
    if (s.rank == 0) base = 0;
    DeviceBuffer<unsigned long long> nonzero;
    if (validate) {
        nonzero.allocate(1);
        checkCuda(cudaMemset(nonzero.data, 0, sizeof(s.nonZeroKij)));
    }
    // Nonblocking worker streams consume geometry and the optional zeroed counter.
    checkCuda(cudaDeviceSynchronize());

    // Workers own consecutive rows and independent streams. Bound pinned memory
    // per worker and overlap CPU sampling with other workers' GPU traversal.
    #pragma omp parallel num_threads(workers)
    {
        checkCuda(cudaSetDevice(s.device));
        int tid = omp_get_thread_num(), threads = omp_get_num_threads();
        int first = size_t(s.rows) * tid / threads;
        int last = size_t(s.rows) * (tid + 1) / threads;
        int capacity = std::min<size_t>(batchSize, size_t(last - first) * n);
        if (capacity) {
            SampleStream rng(base + offsets[first]);
            cudaStream_t stream;
            checkCuda(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
            cudaEvent_t copied;
            checkCuda(cudaEventCreateWithFlags(&copied, cudaEventDisableTiming));
            DeviceBuffer<float4> samples;
            samples.allocate(capacity * NUM_RAYS);
            float4* host;
            checkCuda(cudaMallocHost(&host, capacity * NUM_RAYS * sizeof(float4)));
            size_t end = size_t(last) * n;
            for (size_t start = size_t(first) * n; start < end; start += capacity) {
                int pairs = std::min<size_t>(capacity, end - start);
                for (int pair = 0; pair < pairs; ++pair) {
                    size_t entry = start + pair;
                    bool visible = active(s.first + entry / n, entry % n);
                    for (int r = 0; r < NUM_RAYS; ++r) {
                        float4& sample = host[pair * NUM_RAYS + r];
                        if (visible) sample = {rng.next(), rng.next(), rng.next(), rng.next()};
                        else sample = {-1, 0, 0, 0};
                    }
                }
                checkCuda(cudaMemcpyAsync(samples.data, host, pairs * NUM_RAYS * sizeof(float4),
                                          cudaMemcpyHostToDevice, stream));
                checkCuda(cudaEventRecord(copied, stream));
                formFactors<<<(pairs * NUM_RAYS + blockSize - 1) / blockSize, blockSize, 0, stream>>>(
                    s.geometry.data, s.nodes.data, s.indices.data, s.areas.data, samples.data,
                    n, s.first, start, pairs, s.weights.data, s.delays.data, nonzero.data);
                checkCuda(cudaGetLastError());
                checkCuda(cudaEventSynchronize(copied));
            }
            checkCuda(cudaStreamSynchronize(stream));
            checkCuda(cudaEventDestroy(copied));
            checkCuda(cudaFreeHost(host));
            checkCuda(cudaStreamDestroy(stream));
        }
    }
    if (validate) {
        unsigned long long local;
        checkCuda(cudaMemcpy(&local, nonzero.data, sizeof(local), cudaMemcpyDeviceToHost));
        MPI_Reduce(&local, &s.nonZeroKij, 1, MPI_UNSIGNED_LONG_LONG, MPI_SUM, 0, MPI_COMM_WORLD);
    }
}

__global__ void waveStep(int n, int first, int step, int source, int steps, float rho,
                         const float* weights, const unsigned char* delays, float* recent, float* history) {
    __shared__ float terms[blockSize];
    int row = blockIdx.x;
    size_t start = size_t(row) * n;
    float sum = ZERO;
    for (int tile = 0; tile < n; tile += blockSize) {
        int j = tile + threadIdx.x;
        float term = ZERO;
        if (j < n && j != first + row) {
            int lag = delays[start + j];
            float weight = weights[start + j];
            if (step >= lag && weight > ZERO) {
                float b = recent[size_t((step - lag) % historySlots) * n + j];
                if (!(b <= ZERO)) term = weight * b;
            }
        }
        terms[threadIdx.x] = term;
        __syncthreads();
        if (!threadIdx.x)
            for (int k = 0; k < min(blockSize, n - tile); ++k) sum += terms[k];
        __syncthreads();
    }
    if (!threadIdx.x) {
        int i = first + row;
        float emission = (i == source && step < steps / 2) ? ONE : ZERO;
        float radiosity = rho * sum + emission;
        recent[size_t(step % historySlots) * n + i] = radiosity;
        history[size_t(row) * steps + step] = radiosity;
    }
}

void propagate(SimulationState& s, float reflectivity) {
    int n = s.numTriangles;
    float* slice = nullptr;
    if (s.ranks > 1) checkCuda(cudaMallocHost(&slice, n * sizeof(float)));
    for (int t = 0; t < static_cast<int>(s.numTimesteps); ++t) {
        if (s.rows) {
            waveStep<<<s.rows, blockSize>>>(n, s.first, t, s.sourceIndex, s.numTimesteps,
                reflectivity, s.weights.data, s.delays.data, s.recent.data, s.history.data);
            checkCuda(cudaGetLastError());
        }
        if (slice) {
            float* device = s.recent.data + size_t(t % historySlots) * n;
            if (s.rows) checkCuda(cudaMemcpy(slice + s.first, device + s.first,
                                    s.rows * sizeof(float), cudaMemcpyDeviceToHost));
            MPI_Allgatherv(MPI_IN_PLACE, 0, MPI_FLOAT, slice, s.counts.data(),
                           s.offsets.data(), MPI_FLOAT, MPI_COMM_WORLD);
            checkCuda(cudaMemcpy(device, slice, n * sizeof(float), cudaMemcpyHostToDevice));
        }
    }
    checkCuda(cudaDeviceSynchronize());
    if (slice) checkCuda(cudaFreeHost(slice));
}

__global__ void correlations(int steps, const float* source, const float* history,
                             float* distances, int* energy) {
    int row = blockIdx.x;
    float best = ZERO;
    int lag = 0;
    bool received = false;
    for (int t = threadIdx.x; t < steps; t += blockSize) {
        float sum = ZERO;
        for (int tt = t; tt < steps; ++tt)
            sum += source[tt - t] * history[size_t(row) * steps + tt];
        if (sum > best) {
            best = sum;
            lag = t;
        }
        if (energy && history[size_t(row) * steps + t] > EPSILON) received = true;
    }
    __shared__ float scores[blockSize];
    __shared__ int lags[blockSize];
    scores[threadIdx.x] = best;
    lags[threadIdx.x] = lag;
    int anyEnergy = __syncthreads_or(received);
    if (!threadIdx.x) {
        for (int k = 1; k < blockSize; ++k) {
            if (scores[k] > best || (scores[k] == best && lags[k] < lag)) {
                best = scores[k];
                lag = lags[k];
            }
        }
        distances[row] = WAVE_SPEED * lag;
        if (anyEnergy) atomicAdd(energy, 1);
    }
}

void computeDistances(SimulationState& s, bool validate) {
    DeviceBuffer<float> result;
    DeviceBuffer<int> energy;
    result.allocate(s.rows);
    if (validate) {
        energy.allocate(1);
        checkCuda(cudaMemset(energy.data, 0, sizeof(int)));
    }
    DeviceBuffer<float> sourceCopy;
    const float* source = nullptr;
    if (s.numTimesteps) {
        int owner = std::upper_bound(s.offsets.begin(), s.offsets.end(), int(s.sourceIndex))
                    - s.offsets.begin() - 1;
        if (s.rank == owner)
            source = s.history.data + (s.sourceIndex - s.first) * s.numTimesteps;
        if (s.ranks > 1) {
            std::vector<float> trace(s.numTimesteps);
            if (s.rank == owner)
                checkCuda(cudaMemcpy(trace.data(), source, trace.size() * sizeof(float),
                                     cudaMemcpyDeviceToHost));
            MPI_Bcast(trace.data(), trace.size(), MPI_FLOAT, owner, MPI_COMM_WORLD);
            if (s.rank != owner) {
                sourceCopy.upload(trace);
                source = sourceCopy.data;
            }
        }
    }
    std::vector<float> local(s.rows);
    if (s.rows) {
        correlations<<<s.rows, blockSize>>>(s.numTimesteps, source, s.history.data,
                                             result.data, energy.data);
        checkCuda(cudaMemcpy(local.data(), result.data, s.rows * sizeof(float), cudaMemcpyDeviceToHost));
    }
    if (validate) {
        int count;
        checkCuda(cudaMemcpy(&count, energy.data, sizeof(int), cudaMemcpyDeviceToHost));
        MPI_Reduce(&count, &s.receivedEnergy, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
    }
    if (!s.rank) s.distances.resize(s.numTriangles);
    MPI_Gatherv(local.data(), s.rows, MPI_FLOAT, s.distances.data(), s.counts.data(),
                s.offsets.data(), MPI_FLOAT, 0, MPI_COMM_WORLD);
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
    int receivedEnergy = state.receivedEnergy;
    printf("  Triangles receiving energy: %d/%zu\n", receivedEnergy, state.numTriangles);

    if (receivedEnergy == 0) {
        printf("  ERROR: No triangles received energy - simulation failed\n");
        return false;
    }

    // Check Kij matrix (should have some non-zero entries)
    unsigned long long nonZeroKij = state.nonZeroKij;
    printf("  Non-zero form factors: %llu/%zu (%.2f%%)\n",
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

int main(int argc, char** argv) {
    int targetTriangles = 320;
    int timesteps = 50;
    int sourceIdx = 0;
    val_t reflectivity = 0.8f;
    bool validate = false;
    bool printResults = false;

    int provided, rank, ranks;
    MPI_Init_thread(&argc, &argv, MPI_THREAD_FUNNELED, &provided);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &ranks);
    if (provided < MPI_THREAD_FUNNELED) MPI_Abort(MPI_COMM_WORLD, 1);

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
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) printf("Unknown option: %s\n", argv[i]);
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 1;
        }
    }

    int subdivisions = getSubdivisionsForTriangleCount(targetTriangles);

    if (timesteps < 0) {
        if (rank == 0) fprintf(stderr, "Timesteps must be non-negative\n");
        MPI_Finalize();
        return 1;
    }
    if (rank == 0) {
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
    initialize(state, subdivisions, timesteps, sourceIdx, rank, ranks);
    MPI_Barrier(MPI_COMM_WORLD);
    double start = MPI_Wtime();
    precompute(state, validate);
    MPI_Barrier(MPI_COMM_WORLD);
    double preEnd = MPI_Wtime();
    propagate(state, reflectivity);
    MPI_Barrier(MPI_COMM_WORLD);
    double simEnd = MPI_Wtime();
    computeDistances(state, validate);
    double distEnd = MPI_Wtime();
    double localDuration = distEnd - start;
    double maxDuration = 0.0;
    MPI_Reduce(&localDuration, &maxDuration, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    int status = 0;
    if (rank == 0) {
        long preDuration = static_cast<long>((preEnd - start) * 1000);
        long simDuration = static_cast<long>((simEnd - preEnd) * 1000);
        long distDuration = static_cast<long>((distEnd - simEnd) * 1000);
        printf("Precomputation time: %ld ms\n", preDuration);
        printf("Simulation time: %ld ms\n", simDuration);
        printf("Distance computation time: %ld ms\n", distDuration);

        // Total time
        long totalTime = static_cast<long>(maxDuration * 1000);
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

        // Aggregate device storage for propagation across all ranks.
        size_t waveBytes = n * n * (sizeof(float) + sizeof(unsigned char)) +
            n * (t + size_t(ranks) * std::min<size_t>(t, historySlots)) * sizeof(float);
        printf("  Memory usage: %.2f MB\n", waveBytes / (1024.0 * 1024.0));

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

        if (validate && !validateResults(state)) status = 1;
    }
    MPI_Bcast(&status, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Finalize();
    return status;
}
