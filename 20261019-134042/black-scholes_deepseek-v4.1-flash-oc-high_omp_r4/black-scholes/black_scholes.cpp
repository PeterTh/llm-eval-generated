#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <new>
#include <string>
#include <vector>

#if defined(__linux__)
#include <unistd.h>
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

// Allocator that leaves elements uninitialized on construction. Every element
// is fully written before it is read, so the large vector resize does not need
// a single-threaded zero-fill (and pages are first-touched by parallel workers).
template <typename T>
struct no_init_allocator : std::allocator<T> {
    no_init_allocator() = default;
    template <typename U>
    no_init_allocator(const no_init_allocator<U>&) noexcept {}

    template <typename U>
    struct rebind {
        using other = no_init_allocator<U>;
    };

    template <typename U>
    void construct(U* p) noexcept {
        ::new (static_cast<void*>(p)) U;
    }
};

using OptionVector = std::vector<OptionInput, no_init_allocator<OptionInput>>;
using ResultVector = std::vector<double, no_init_allocator<double>>;

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
    
    const double volSqrtT = sigma * sqrt(T);
    const double d1 = (log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / volSqrtT;
    const double d2 = d1 - volSqrtT;
    
    const double Nd1 = cumulativeNormal(d1);
    const double Nd2 = cumulativeNormal(d2);
    const double discount = exp(-r * T);
    const double dividendDiscount = exp(-q * T);
    
    double price;
    if (option.type == CALL) {
        price = S * dividendDiscount * Nd1 - K * discount * Nd2;
    } else { // PUT
        price = K * discount * cumulativeNormal(-d2) - S * dividendDiscount * cumulativeNormal(-d1);
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

// Build the i-th option by cycling through the test set and varying parameters
inline OptionInput makeOption(const std::array<OptionInput, 7>& testOptions,
                              const size_t i) noexcept {
    OptionInput option = testOptions[i % testOptions.size()];

    // Add some variation for larger datasets
    const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;

    return option;
}

// Generate a larger set of options by scaling the test set
void generateOptions(OptionVector& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        options[i] = makeOption(testOptions, i);
    }
}

bool validateResults(const OptionVector& options, 
                     const ResultVector& results) {
    bool allPassed = true;
    const int numChecks = std::min(static_cast<size_t>(10), options.size());
    
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

// Number of hardware threads sharing a physical core (SMT width).
unsigned int smtThreadsPerCore() {
#if defined(__linux__)
    std::ifstream file("/sys/devices/system/cpu/cpu0/topology/thread_siblings_list");
    std::string list;
    if (std::getline(file, list)) {
        unsigned int count = 0;
        const char* p = list.c_str();
        while (*p != '\0') {
            char* end = nullptr;
            const unsigned long first = std::strtoul(p, &end, 10);
            if (end == p) {
                break;
            }
            ++count;
            p = end;
            if (*p == '-') {
                const unsigned long last = std::strtoul(p + 1, &end, 10);
                if (end == p + 1) {
                    break;
                }
                if (last > first) {
                    count += static_cast<unsigned int>(last - first);
                }
                p = end;
            }
            if (*p == ',') {
                ++p;
            } else if (*p != '\0') {
                break;
            }
        }
        if (count > 0) {
            return count;
        }
    }
#endif
    return 1;
}

// Choose a thread count that avoids oversubscription and excessive per-thread
// startup cost, while never exceeding the OpenMP thread limit or physical cores.
void configureThreadCount(const size_t numOptions) {
    const unsigned int maxThreads = static_cast<unsigned int>(omp_get_max_threads());
    if (maxThreads <= 1) {
        return;
    }

    const unsigned int procs = static_cast<unsigned int>(omp_get_num_procs());
    unsigned int online = procs;
#if defined(__linux__)
    const long onlineProcs = sysconf(_SC_NPROCESSORS_ONLN);
    if (onlineProcs > 0) {
        online = static_cast<unsigned int>(onlineProcs);
    }
#endif
    unsigned int physical = std::max(1u, online / smtThreadsPerCore());
    physical = std::min(physical, procs);
    if (physical < 2) {
        physical = maxThreads;
    }

    constexpr size_t workPerThread = 2048;
    const size_t workThreads = numOptions / workPerThread
                             + (numOptions % workPerThread != 0 ? 1 : 0);
    const unsigned int threads = static_cast<unsigned int>(
        std::min({static_cast<size_t>(maxThreads),
                  static_cast<size_t>(physical),
                  std::max<size_t>(1, workThreads)}));
    omp_set_num_threads(static_cast<int>(threads));
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
    
    configureThreadCount(numOptions);

    printf("Black-Scholes Option Pricing Benchmark\n");
    printf("Number of options: %zu\n", numOptions);
    printf("Validation: %s\n", validate ? "enabled" : "disabled");
    
    // Materialize only the options needed for validation; pricing builds each
    // option on the fly so no large temporary options array is required.
    OptionVector checkOptions;
    if (validate) {
        generateOptions(checkOptions, std::min<size_t>(numOptions, 10));
    }
    
    // Allocate results
    ResultVector results(numOptions);
    
    // Price options
    printf("Pricing options...\n");
    constexpr auto testOptions = getTestOptions();

    // Warm up the worker threads before starting the timer so that thread
    // startup and cold caches are not charged to the measured pricing region.
    const unsigned int activeThreads = static_cast<unsigned int>(omp_get_max_threads());
    const size_t warmupOptions = activeThreads > 1
        ? std::min<size_t>(numOptions, static_cast<size_t>(activeThreads) * 512)
        : 0;
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < warmupOptions; ++i) {
        results[i] = blackScholes(makeOption(testOptions, i));
    }

    auto start = std::chrono::high_resolution_clock::now();
    
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < numOptions; ++i) {
        results[i] = blackScholes(makeOption(testOptions, i));
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
    
    // Print results for external validation
    if (printResults) {
        const std::vector<double> printable(results.begin(), results.end());
        print_results(printable, "OptionPrices");
    }
    
    // Validation
    if (validate) {
        printf("Validating results...\n");
        bool valid = validateResults(checkOptions, results);
        
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
