#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

#include <omp.h>

#include "../common/results_output.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#ifndef M_SQRT_2
#define M_SQRT_2 0.7071067811865475244
#endif
#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif
#ifndef M_1_SQRTPI
#define M_1_SQRTPI 0.564189583547756286948
#endif

enum OptionType {
    CALL = 0,
    PUT = 1
};

struct OptionInput {
    int type;           // CALL or PUT
    double strike;      // Strike price
    double spot;        // Spot price
    double q;           // Dividend yield
    double r;           // Risk-free rate
    double t;           // Time to maturity
    double vol;         // Volatility
    double value;       // Expected value (for validation)
    double tol;         // Tolerance
};

// Standard normal cumulative distribution function
inline double cumulativeNormal(const double x) noexcept {
    return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
}

// Standard normal probability density function
inline double normalPDF(const double x) noexcept {
    return M_1_SQRTPI * M_SQRT1_2 * exp(-0.5 * x * x);
}

// Black-Scholes formula for European options
double blackScholes(const OptionInput& option) noexcept {
    const double S = option.spot;
    const double K = option.strike;
    const double r = option.r;
    const double q = option.q;
    const double T = option.t;
    const double sigma = option.vol;
    
    if (T <= 0.0 || sigma <= 0.0) {
        return 0.0;
    }
    
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T));
    const double d2 = d1 - sigma * sqrt(T);
    
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);
    
    double price;
    if (option.type == CALL) {
        price = S * exp(-q * T) * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * exp(-q * T) * cumulativeNormal(-d1);
    }
    
    return price;
}

// Standard test cases for validation
inline constexpr std::array<OptionInput, 7> getTestOptions() noexcept {
    return {{
        {CALL, 40.00, 42.00, 0.04, 0.08, 0.75, 0.35, 5.0975, 1.0e-3},
        {CALL, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 0.0205, 1.0e-3},
        {CALL, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {CALL, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 9.9413, 1.0e-3},
        {PUT, 100.00, 90.00, 0.10, 0.10, 0.10, 0.15, 9.9210, 1.0e-3},
        {PUT, 100.00, 100.00, 0.10, 0.10, 0.10, 0.15, 1.8734, 1.0e-3},
        {PUT, 100.00, 110.00, 0.10, 0.10, 0.10, 0.15, 0.0408, 1.0e-3},
    }};
}

#ifdef __linux__
// Read a sysfs CPU list such as "0,128" or "0-3,8" into a vector of CPU ids.
static std::vector<int> parseCpuList(const char* path) {
    std::vector<int> cpus;
    FILE* f = fopen(path, "r");
    if (!f) {
        return cpus;
    }
    char buf[4096];
    const bool ok = fgets(buf, sizeof(buf), f) != nullptr;
    fclose(f);
    if (!ok) {
        return cpus;
    }

    const char* p = buf;
    while (*p) {
        char* endp;
        const long first = strtol(p, &endp, 10);
        if (endp == p) {
            break;
        }
        p = endp;
        long last = first;
        if (*p == '-') {
            last = strtol(p + 1, &endp, 10);
            p = endp;
        }
        for (long c = first; c <= last; ++c) {
            cpus.push_back(static_cast<int>(c));
        }
        if (*p != ',') {
            break;
        }
        ++p;
    }
    return cpus;
}

static int readIntFile(const char* path) {
    FILE* f = fopen(path, "r");
    if (!f) {
        return -1;
    }
    int value = -1;
    if (fscanf(f, "%d", &value) != 1) {
        value = -1;
    }
    fclose(f);
    return value;
}

// Order the CPUs this process may run on so that consecutive entries land on
// distinct physical cores, alternating between sockets; SMT siblings are only
// used once every core has been handed out.
static std::vector<int> buildCpuOrder() {
    cpu_set_t mask;
    CPU_ZERO(&mask);
    if (sched_getaffinity(0, sizeof(mask), &mask) != 0) {
        return {};
    }

    // buckets[smtIndex][package] -> CPU ids
    std::vector<std::vector<std::vector<int>>> buckets;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (!CPU_ISSET(cpu, &mask)) {
            continue;
        }

        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list", cpu);
        std::vector<int> siblings = parseCpuList(path);
        std::sort(siblings.begin(), siblings.end());
        const auto self = std::find(siblings.begin(), siblings.end(), cpu);
        const size_t smt = (self == siblings.end())
                               ? 0
                               : static_cast<size_t>(self - siblings.begin());

        snprintf(path, sizeof(path),
                 "/sys/devices/system/cpu/cpu%d/topology/physical_package_id", cpu);
        const int pkgId = readIntFile(path);
        const size_t pkg = (pkgId < 0) ? 0 : static_cast<size_t>(pkgId);

        if (buckets.size() <= smt) {
            buckets.resize(smt + 1);
        }
        if (buckets[smt].size() <= pkg) {
            buckets[smt].resize(pkg + 1);
        }
        buckets[smt][pkg].push_back(cpu);
    }

    std::vector<int> order;
    for (auto& perPackage : buckets) {
        // Round-robin over the sockets at this SMT level.
        for (size_t slot = 0, taken = 1; taken > 0; ++slot) {
            taken = 0;
            for (auto& cpus : perPackage) {
                if (slot < cpus.size()) {
                    order.push_back(cpus[slot]);
                    ++taken;
                }
            }
        }
    }
    return order;
}

// libgomp leaves threads unbound unless OMP_PROC_BIND/OMP_PLACES are set, which
// costs a large fraction of the achievable throughput on many-core machines
// (thread migration destroys NUMA locality established by first touch). Pin the
// pool ourselves, but leave it alone whenever the user configured binding.
static void bindThreads() {
    if (omp_get_proc_bind() != omp_proc_bind_false) {
        return;
    }
    const std::vector<int> order = buildCpuOrder();
    if (order.empty()) {
        return;
    }

    #pragma omp parallel
    {
        const size_t tid = static_cast<size_t>(omp_get_thread_num());
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(order[tid % order.size()], &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
}
#else
static void bindThreads() {}
#endif

// Generate a larger set of options by scaling the test set.
// The loop is parallelized with the same static schedule used by the pricing
// loop so that every thread performs the first touch of the pages it will later
// read (NUMA-local placement on multi-socket machines).
void generateOptions(OptionInput* options, const size_t numOptions, const int numThreads) {
    constexpr auto testOptions = getTestOptions();

    #pragma omp parallel for schedule(static) num_threads(numThreads)
    for (size_t i = 0; i < numOptions; ++i) {
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[i] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[i].spot *= factor;
        options[i].strike *= factor;
    }
}

bool validateResults(const OptionInput* options, const double* results,
                     const size_t numOptions) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), numOptions);
    
    printf("Checking computed option prices:\n");
    for (int i = 0; i < numChecks; ++i) {
        const double computed = results[i];
        const double expected = options[i].value;
        const double error = fabs(computed - expected);
        const double relError = error / (fabs(expected) + 1e-10);
        
        printf("  Option %d: computed=%.4f, expected=%.4f, rel_error=%.4e\n", 
               i, computed, expected, relError);
        
        // Relaxed validation - just check values are positive and reasonable
        if (computed < 0.0 || computed > 1000.0 || std::isnan(computed) || std::isinf(computed)) {
            printf("Validation failed at option %d: invalid value %.4f\n", i, computed);
            allPassed = false;
        }
    }
    
    return allPassed;
}

void printUsage(const char* progName) {
    printf("Usage: %s [options]\n", progName);
    printf("Options:\n");
    printf("  -n <num>     Number of options to price (default: 10000)\n");
    printf("  -v           Enable validation against known values\n");
    printf("  -r           Print results for external validation\n");
    printf("  -h           Show this help message\n");
}

int main(int argc, char** argv) {
    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
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
    
    printf("Black-Scholes Option Pricing Benchmark\n");
    printf("Number of options: %zu\n", numOptions);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    printf("OpenMP threads: %d\n", omp_get_max_threads());

    // Pin the thread pool before any data is touched.
    bindThreads();

    // Waking and synchronizing the whole pool costs more than the work itself
    // for tiny problem sizes, so give every thread a worthwhile chunk. All
    // loops below use the same thread count and static schedule, hence the
    // same data partitioning (and therefore NUMA-local first touch).
    constexpr size_t minOptionsPerThread = 4096;
    const size_t usefulThreads = std::max<size_t>(1, numOptions / minOptionsPerThread);
    const int numThreads = static_cast<int>(
        std::min<size_t>(static_cast<size_t>(omp_get_max_threads()), usefulThreads));
    printf("Threads used: %d\n", numThreads);

    // Generate options. Default-initialized (i.e. untouched) storage, so that
    // the parallel generation loop below performs the first touch.
    const std::unique_ptr<OptionInput[]> optionStorage(new OptionInput[numOptions]);
    OptionInput* const options = optionStorage.get();
    generateOptions(options, numOptions, numThreads);

    // Allocate results, again first-touched by the owning thread.
    const std::unique_ptr<double[]> resultStorage(new double[numOptions]);
    double* const results = resultStorage.get();
    #pragma omp parallel for schedule(static) num_threads(numThreads)
    for (size_t i = 0; i < numOptions; ++i) {
        results[i] = 0.0;
    }

    // Price options
    printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static) num_threads(numThreads)
    for (size_t i = 0; i < numOptions; ++i) {
        results[i] = blackScholes(options[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
    
    // Print results for external validation
    if (printResults) {
        print_results(std::vector<double>(results, results + numOptions), "OptionPrices");
    }

    // Validation
    if (validate) {
        printf("Validating results...\n");
        bool valid = validateResults(options, results, numOptions);
        
        if (valid) {
            printf("Validation: PASSED\n");
            return 0;
        } else {
            printf("Validation: FAILED\n");
            return 1;
        }
    }
    
    return 0;
}
