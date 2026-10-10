/**
 * Room Response Simulation Benchmark
 * 
 * This is a CUDA-parallel implementation of room impulse response
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
#include <vector>
#include <cuda_runtime.h>

#define HD __host__ __device__

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

    HD constexpr Vec3() : x(0), y(0), z(0) {}
    HD constexpr Vec3(val_t x, val_t y, val_t z) : x(x), y(y), z(z) {}
    HD explicit constexpr Vec3(val_t v) : x(v), y(v), z(v) {}

    HD Vec3 operator+(const Vec3& o) const { return {x + o.x, y + o.y, z + o.z}; }
    HD Vec3 operator-(const Vec3& o) const { return {x - o.x, y - o.y, z - o.z}; }
    HD Vec3 operator*(val_t s) const { return {x * s, y * s, z * s}; }
    HD Vec3 operator/(val_t s) const { return {x / s, y / s, z / s}; }
    HD Vec3 operator-() const { return {-x, -y, -z}; }

    HD val_t dot(const Vec3& o) const { return x * o.x + y * o.y + z * o.z; }
    HD Vec3 cross(const Vec3& o) const {
        return {y * o.z - z * o.y, z * o.x - x * o.z, x * o.y - y * o.x};
    }

    HD val_t squaredNorm() const { return x * x + y * y + z * z; }
    HD val_t norm() const { return std::sqrt(squaredNorm()); }
    HD Vec3 normalized() const {
        val_t n = norm();
        return n > EPSILON ? *this / n : Vec3();
    }

    HD bool operator==(const Vec3& o) const {
        return std::abs(x - o.x) < EPSILON && std::abs(y - o.y) < EPSILON && std::abs(z - o.z) < EPSILON;
    }
};

struct Triangle {
    Vec3 a, b, c;
    Vec3 _normal;

    HD Triangle() = default;
    HD Triangle(const Vec3& a, const Vec3& b, const Vec3& c)
        : a(a), b(b), c(c), _normal((b - a).cross(c - a).normalized()) {}

    HD Vec3 center() const { return (a + b + c) / 3.0f; }
    HD Vec3 normal() const { return _normal; }

    HD val_t area() const {
        Vec3 ab = b - a;
        Vec3 ac = c - a;
        return 0.5f * ab.cross(ac).norm();
    }

    HD bool operator==(const Triangle& o) const { return a == o.a && b == o.b && c == o.c; }
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

// ============================================================================
// Ray-Triangle Intersection (Möller-Trumbore algorithm)
// ============================================================================

HD val_t rayTriangleIntersect(const Vec3& orig, const Vec3& dir,
                            const Vec3& v0, const Vec3& v1, const Vec3& v2) {
    Vec3 e1 = v1 - v0;
    Vec3 e2 = v2 - v0;
    Vec3 pvec = dir.cross(e2);
    val_t det = e1.dot(pvec);

    if (std::abs(det) < EPSILON) return 0x1.fffffep127f;

    val_t invDet = 1.0f / det;
    Vec3 tvec = orig - v0;
    val_t u = tvec.dot(pvec) * invDet;
    if (u < 0.0f || u > 1.0f) return 0x1.fffffep127f;

    Vec3 qvec = tvec.cross(e1);
    val_t v = dir.dot(qvec) * invDet;
    if (v < 0.0f || u + v > 1.0f) return 0x1.fffffep127f;

    return e2.dot(qvec) * invDet;
}

// ============================================================================
// Form Factor Computation (Kij)
// ============================================================================

// Compute the cosine of angle between vector and triangle normal
HD val_t cosPhi(const Vec3& v, const Vec3& normal) {
    val_t vNorm = v.norm();
    if (vNorm <= EPSILON) return ZERO;
    return fmaxf(ZERO, v.dot(normal) / vNorm);
}

// ============================================================================
// Tau (time delay) Computation
// ============================================================================

HD int computeTau(const Triangle& triI, const Triangle& triJ) {
    val_t dist = (triI.center() - triJ.center()).norm();
    return static_cast<int>(std::ceil(dist * INV_WAVE_SPEED));
}


// CUDA storage and kernels. Flattening the existing octree preserves its exact
// traversal and overlap tests, without device recursion or per-ray stacks.
static void cudaCheck(cudaError_t error) {
    if (error != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s\n", cudaGetErrorString(error));
        std::exit(EXIT_FAILURE);
    }
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    DeviceBuffer() = default;
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    ~DeviceBuffer() { if (data) cudaFree(data); }
    void allocate(size_t count) {
        if (count) cudaCheck(cudaMalloc(reinterpret_cast<void**>(&data), count * sizeof(T)));
    }
    void upload(const T* src, size_t count) {
        if (count) cudaCheck(cudaMemcpy(data, src, count * sizeof(T), cudaMemcpyHostToDevice));
    }
    void download(T* dst, size_t count) const {
        if (count) cudaCheck(cudaMemcpy(dst, data, count * sizeof(T), cudaMemcpyDeviceToHost));
    }
};

struct DeviceNode {
    Vec3 center, halfExtent;
    size_t begin, count, escape;
    __device__ bool intersects(const Vec3& p1, const Vec3& p2) const {
        Vec3 d = (p2 - p1) * 0.5f;
        Vec3 c = p1 + d - center;
        Vec3 ad = {fabsf(d.x), fabsf(d.y), fabsf(d.z)};
        if (fabsf(c.x) > halfExtent.x + ad.x) return false;
        if (fabsf(c.y) > halfExtent.y + ad.y) return false;
        if (fabsf(c.z) > halfExtent.z + ad.z) return false;
        if (fabsf(d.y*c.z-d.z*c.y) > halfExtent.y*ad.z+halfExtent.z*ad.y+EPSILON) return false;
        if (fabsf(d.z*c.x-d.x*c.z) > halfExtent.z*ad.x+halfExtent.x*ad.z+EPSILON) return false;
        if (fabsf(d.x*c.y-d.y*c.x) > halfExtent.x*ad.y+halfExtent.y*ad.x+EPSILON) return false;
        return true;
    }
};

static void flattenOctree(const Octree& tree, std::vector<DeviceNode>& nodes,
                          std::vector<size_t>& indices) {
    size_t index = nodes.size();
    nodes.push_back({tree.center, tree.halfExtent, indices.size(), tree.triangleIndices.size(), 0});
    indices.insert(indices.end(), tree.triangleIndices.begin(), tree.triangleIndices.end());
    for (const auto& child : tree.children)
        if (child) flattenOctree(*child, nodes, indices);
    nodes[index].escape = nodes.size();
}

__device__ bool deviceBlocked(const Vec3& from, const Vec3& to, const Triangle* triangles,
                             const DeviceNode* nodes, const size_t* indices,
                             size_t nodeCount, size_t src, size_t dst) {
    Vec3 dir = to - from;
    float length = dir.norm();
    if (length < EPSILON) return true;
    Vec3 normal = dir / length;
    for (size_t n = 0; n < nodeCount;) {
        const DeviceNode& node = nodes[n];
        if (n != 0 && !node.intersects(from, to)) { n = node.escape; continue; }
        for (size_t k = 0; k < node.count; ++k) {
            size_t index = indices[node.begin + k];
            if (index == src || index == dst) continue;
            const Triangle& tri = triangles[index];
            float distance = rayTriangleIntersect(from, normal, tri.a, tri.b, tri.c);
            if (distance > EPSILON && distance < length - EPSILON) return true;
        }
        ++n;
    }
    return false;
}

constexpr int PAIR_BATCH = 65536;
constexpr int DRAWS_PER_PAIR = NUM_RAYS * 4;

// Parallel prefix scan assigns each non-culled pair its original position in
// the single MT19937 stream. A cooperative twist produces exactly the same
// draws as std::mt19937, including its in-place recurrence and float rounding.
__global__ void prepareSamples(const Triangle* triangles, size_t n, size_t base,
                               size_t pairs, int* offsets, float* samples, uint32_t* state) {
    __shared__ unsigned scan[256];
    __shared__ uint32_t mt[624], old[624];
    __shared__ unsigned total;
    unsigned tid = threadIdx.x;
    constexpr unsigned PER_THREAD = PAIR_BATCH / 256;
    unsigned count = 0;
    // Contiguous local ranges followed by a block scan preserve global pair
    // ordering with only 256 shared scan entries, regardless of batch size.
    for (unsigned k = tid * PER_THREAD; k < (tid+1) * PER_THREAD; ++k) {
        size_t pair = base + k;
        offsets[k] = count;
        count += pair < pairs && pair / n != pair % n &&
                 triangles[pair/n].normal().dot(triangles[pair%n].normal()) <= 0.99f;
    }
    scan[tid] = count;
    __syncthreads();
    for (unsigned stride = 1; stride < 256; stride *= 2) {
        unsigned k = (tid+1) * stride * 2 - 1;
        if (k < 256) scan[k] += scan[k-stride];
        __syncthreads();
    }
    if (tid == 0) { total = scan[255] * DRAWS_PER_PAIR; scan[255] = 0; }
    __syncthreads();
    for (unsigned stride = 128; stride; stride /= 2) {
        unsigned k = (tid+1) * stride * 2 - 1;
        if (k < 256) {
            unsigned left = scan[k-stride];
            scan[k-stride] = scan[k];
            scan[k] += left;
        }
        __syncthreads();
    }
    for (unsigned k = tid * PER_THREAD; k < (tid+1) * PER_THREAD; ++k)
        offsets[k] = (offsets[k] + scan[tid]) * DRAWS_PER_PAIR;
    for (unsigned k = tid; k < 624; k += blockDim.x) mt[k] = state[k];
    __syncthreads();
    unsigned position = state[624];
    unsigned written = 0;
    while (written < total) {
        if (position == 624) {
            for (unsigned k = tid; k < 624; k += blockDim.x) old[k] = mt[k];
            __syncthreads();
            for (unsigned first = 0; first < 623; first += 227) {
                unsigned last = min(first + 227, 623u);
                for (unsigned k = first + tid; k < last; k += blockDim.x) {
                    uint32_t y = (old[k] & 0x80000000u) | (old[k+1] & 0x7fffffffu);
                    mt[k] = (k < 227 ? old[k+397] : mt[k-227]) ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0u);
                }
                __syncthreads();
            }
            if (tid == 0) {
                uint32_t y = (old[623] & 0x80000000u) | (mt[0] & 0x7fffffffu);
                mt[623] = mt[396] ^ (y >> 1) ^ ((y & 1) ? 0x9908b0dfu : 0u);
            }
            __syncthreads();
            position = 0;
        }
        unsigned count = min(624u - position, total - written);
        for (unsigned k = tid; k < count; k += blockDim.x) {
            uint32_t y = mt[position+k];
            y ^= y >> 11;
            y ^= (y << 7) & 0x9d2c5680u;
            y ^= (y << 15) & 0xefc60000u;
            y ^= y >> 18;
            samples[written+k] = fminf(static_cast<float>(y) * 0x1p-32f, 0x1.fffffep-1f);
        }
        __syncthreads();
        written += count;
        position += count;
    }
    for (unsigned k = tid; k < 624; k += blockDim.x) state[k] = mt[k];
    if (tid == 0) state[624] = position;
}

__device__ Vec3 sampledPoint(const Triangle& t, float u, float v) {
    if (u + v > 1.0f) { u = 1.0f-u; v = 1.0f-v; }
    return t.a + (t.b-t.a)*u + (t.c-t.a)*v;
}

__global__ void formFactorKernel(const Triangle* triangles, const DeviceNode* nodes,
                                const size_t* indices, size_t nodeCount, size_t n,
                                size_t base, size_t pairs, const int* offsets,
                                const float* samples, float* kij) {
    unsigned k = blockIdx.x * blockDim.x + threadIdx.x;
    size_t pair = base + k;
    if (k >= PAIR_BATCH || pair >= pairs) return;
    size_t i = pair / n, j = pair % n;
    const Triangle& a = triangles[i];
    const Triangle& b = triangles[j];
    float value = ZERO;
    if (i != j && a.normal().dot(b.normal()) <= 0.99f) {
        const float* draws = samples + offsets[k];
        for (int r = 0; r < NUM_RAYS; ++r) {
            Vec3 p = sampledPoint(a, draws[4*r], draws[4*r+1]);
            Vec3 q = sampledPoint(b, draws[4*r+2], draws[4*r+3]);
            if (deviceBlocked(p, q, triangles, nodes, indices, nodeCount, i, j)) continue;
            Vec3 v = q-p;
            float squared = v.squaredNorm();
            if (squared < EPSILON) continue;
            float ca = cosPhi(v, a.normal()), cb = cosPhi(-v, b.normal());
            if (ca <= ZERO || cb <= ZERO) continue;
            value += (ca * cb) / (PI * squared);
        }
    }
    // Column-major matrices coalesce accesses across receiving triangles.
    kij[j*n+i] = value * INV_NUM_RAYS;
}

__global__ void delayKernel(const Triangle* triangles, size_t n, int* tau) {
    size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (k < n*n) {
        size_t i = k % n, j = k / n;
        tau[k] = i == j ? 0 : computeTau(triangles[i], triangles[j]);
    }
}

__global__ void propagationKernel(size_t n, size_t t, size_t timesteps, size_t source,
                                  const float* kij, const int* tau, const float* areas,
                                  const float* rho, float* rad) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float sum = ZERO;
    // Keep the original ascending-j accumulation order. All non-self delays
    // are positive on this mesh; a kernel boundary orders successive timesteps.
    for (size_t j = 0; j < n; ++j) {
        if (i == j) continue;
        int delay = tau[j*n+i];
        if (t < static_cast<size_t>(delay)) continue;
        float factor = kij[j*n+i];
        if (factor <= ZERO) continue;
        float value = rad[(t-delay)*n+j];
        if (value <= ZERO) continue;
        sum += fminf(factor*areas[j], ONE) * value;
    }
    rad[t*n+i] = rho[i] * sum + ((i == source && t < timesteps/2) ? ONE : ZERO);
}

__global__ void correlationKernel(size_t n, size_t timesteps, size_t source,
                                  const float* rad, float* correlations) {
    for (size_t k = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         k < n*timesteps; k += size_t(gridDim.x)*blockDim.x) {
        size_t i = k % n, lag = k / n;
        float sum = ZERO;
        for (size_t tt = lag; tt < timesteps; ++tt)
            sum += rad[(tt-lag)*n+source] * rad[tt*n+i];
        correlations[k] = sum;
    }
}

__global__ void distanceKernel(size_t n, size_t timesteps, const float* correlations, float* distances) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float best = ZERO;
    size_t lag = 0;
    for (size_t t = 0; t < timesteps; ++t) {
        float value = correlations[t*n+i];
        if (value > best) { best = value; lag = t; }
    }
    distances[i] = WAVE_SPEED * static_cast<float>(lag);
}

struct DeviceSimulation {
    DeviceBuffer<Triangle> triangles;
    DeviceBuffer<DeviceNode> nodes;
    DeviceBuffer<size_t> indices;
    DeviceBuffer<float> areas, rho, kij, rad, distances, samples, correlations;
    DeviceBuffer<int> tau, offsets;
    DeviceBuffer<uint32_t> rng;
    size_t nodeCount = 0;
};

// ============================================================================
// Room Response Simulation State
// ============================================================================

struct SimulationState {
    size_t numTriangles;
    size_t numTimesteps;

    std::vector<Triangle> triangles;
    std::vector<val_t> areas;       // Area of each triangle
    std::vector<val_t> rho;         // Reflectivity (0.0 to 1.0)
    std::vector<val_t> kij;         // Optional validation copy, column-major
    std::vector<val_t> radB;        // Reflected radiosity (T x N matrix)
    std::vector<val_t> distances;   // Computed distances from source

    Octree octree;                  // Spatial acceleration structure
    DeviceSimulation device;

    size_t sourceIndex;

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
    for (size_t i = 0; i < state.numTriangles; ++i) {
        state.areas[i] = state.triangles[i].area();
    }

    // Initialize reflectivity
    state.rho.resize(state.numTriangles, reflectivity);

    state.distances.resize(state.numTriangles, ZERO);
}

// ============================================================================
// Precomputation Phase
// ============================================================================

void initializeDevice(SimulationState& state) {
    auto& d = state.device;
    size_t n = state.numTriangles, t = state.numTimesteps;
    std::vector<DeviceNode> nodes;
    std::vector<size_t> indices;
    flattenOctree(state.octree, nodes, indices);
    d.nodeCount = nodes.size();
    d.triangles.allocate(n); d.triangles.upload(state.triangles.data(), n);
    d.nodes.allocate(nodes.size()); d.nodes.upload(nodes.data(), nodes.size());
    d.indices.allocate(indices.size()); d.indices.upload(indices.data(), indices.size());
    d.areas.allocate(n); d.areas.upload(state.areas.data(), n);
    d.rho.allocate(n); d.rho.upload(state.rho.data(), n);
    d.kij.allocate(n*n); d.tau.allocate(n*n);
    d.rad.allocate(t*n);
    if (t) cudaCheck(cudaMemset(d.rad.data, 0, t*n*sizeof(float)));
    d.distances.allocate(n); d.correlations.allocate(t*n);
    d.samples.allocate(2 * PAIR_BATCH * DRAWS_PER_PAIR);
    d.offsets.allocate(2 * PAIR_BATCH); d.rng.allocate(625);
    uint32_t mt[625];
    mt[0] = 42;
    for (uint32_t i = 1; i < 624; ++i) mt[i] = 1812433253u * (mt[i-1] ^ (mt[i-1] >> 30)) + i;
    mt[624] = 624;
    d.rng.upload(mt, 625);
}

void computeFormFactors(SimulationState& state) {
    printf("Computing form factors (Kij) with CUDA...\n");
    auto& d = state.device;
    size_t n = state.numTriangles;
    cudaStream_t sampling, visibility;
    cudaCheck(cudaStreamCreate(&sampling));
    cudaCheck(cudaStreamCreate(&visibility));
    cudaEvent_t ready[2], consumed[2];
    for (int k = 0; k < 2; ++k) {
        cudaCheck(cudaEventCreateWithFlags(&ready[k], cudaEventDisableTiming));
        cudaCheck(cudaEventCreateWithFlags(&consumed[k], cudaEventDisableTiming));
    }
    // Double buffering overlaps the deterministic random stream with the much
    // wider ray workload. Events prevent reuse until every reader has finished.
    for (size_t base = 0, batch = 0; base < n*n; base += PAIR_BATCH, ++batch) {
        unsigned slot = batch % 2;
        int* offsets = d.offsets.data + slot * PAIR_BATCH;
        float* samples = d.samples.data + slot * PAIR_BATCH * DRAWS_PER_PAIR;
        if (batch >= 2) cudaCheck(cudaStreamWaitEvent(sampling, consumed[slot], 0));
        prepareSamples<<<1, 256, 0, sampling>>>(d.triangles.data, n, base, n*n, offsets, samples, d.rng.data);
        cudaCheck(cudaEventRecord(ready[slot], sampling));
        cudaCheck(cudaStreamWaitEvent(visibility, ready[slot], 0));
        formFactorKernel<<<PAIR_BATCH/128, 128, 0, visibility>>>(d.triangles.data, d.nodes.data,
            d.indices.data, d.nodeCount, n, base, n*n, offsets, samples, d.kij.data);
        cudaCheck(cudaEventRecord(consumed[slot], visibility));
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
    for (int k = 0; k < 2; ++k) {
        cudaCheck(cudaEventDestroy(ready[k]));
        cudaCheck(cudaEventDestroy(consumed[k]));
    }
    cudaCheck(cudaStreamDestroy(sampling));
    cudaCheck(cudaStreamDestroy(visibility));
}

void computeTimeDelays(SimulationState& state) {
    printf("Computing time delays (Tau) with CUDA...\n");
    size_t n = state.numTriangles;
    delayKernel<<<(n*n+255)/256, 256>>>(state.device.triangles.data, n, state.device.tau.data);
    cudaCheck(cudaGetLastError());
}

void runSimulation(SimulationState& state) {
    printf("Running wave propagation simulation with CUDA...\n");
    auto& d = state.device;
    size_t n = state.numTriangles;
    for (size_t t = 0; t < state.numTimesteps; ++t) {
        propagationKernel<<<(n+127)/128, 128>>>(n, t, state.numTimesteps, state.sourceIndex,
            d.kij.data, d.tau.data, d.areas.data, d.rho.data, d.rad.data);
    }
    cudaCheck(cudaGetLastError());
    cudaCheck(cudaDeviceSynchronize());
}

void computeDistances(SimulationState& state) {
    printf("Computing distances via cross-correlation with CUDA...\n");
    auto& d = state.device;
    size_t n = state.numTriangles, t = state.numTimesteps;
    if (t) correlationKernel<<<std::min(size_t(65535), (n*t+127)/128), 128>>>(
        n, t, state.sourceIndex, d.rad.data, d.correlations.data);
    distanceKernel<<<(n+255)/256, 256>>>(n, t, d.correlations.data, d.distances.data);
    cudaCheck(cudaGetLastError());
    d.distances.download(state.distances.data(), n);
}

// ============================================================================
// Validation
// ============================================================================

bool validateResults(SimulationState& state) {
    state.kij.resize(state.numTriangles * state.numTriangles);
    state.radB.resize(state.numTriangles * state.numTimesteps);
    state.device.kij.download(state.kij.data(), state.kij.size());
    state.device.rad.download(state.radB.data(), state.radB.size());
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
    int nonZeroKij = 0;
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
    initializeSimulation(state, subdivisions, static_cast<size_t>(timesteps),
                         static_cast<size_t>(sourceIdx), reflectivity);
    initializeDevice(state);

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
    // Radiosity history and correlation scratch replace the original two
    // radiosity arrays; sampling storage is bounded independently of mesh size.
    size_t memSampling = 2 * PAIR_BATCH * (DRAWS_PER_PAIR * sizeof(val_t) + sizeof(int));
    size_t totalMem = memKij + memTau + memRad + memSampling;
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
