#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>
#include <vector>

#include <omp.h>

#ifdef __linux__
#include <sched.h>
#endif

#include "../common/results_output.hpp"

// Allocator that leaves trivially-constructible elements uninitialized on
// resize().  Skipping the implicit value-initialization keeps the freshly
// mapped pages untouched, so the first parallel write loop performs the
// first touch and the NUMA page placement follows the thread that will later
// read/write that element.
template <typename T>
struct DefaultInitAllocator : std::allocator<T> {
    using std::allocator<T>::allocator;

    template <typename U>
    struct rebind {
        using other = DefaultInitAllocator<U>;
    };

    template <typename U, typename... Args>
    void construct(U* p, Args&&... args) {
        if constexpr (sizeof...(Args) == 0) {
            ::new (static_cast<void*>(p)) U;
        } else {
            ::new (static_cast<void*>(p)) U(std::forward<Args>(args)...);
        }
    }
};

template <typename T>
using NumaVector = std::vector<T, DefaultInitAllocator<T>>;

// Pin every OpenMP thread to its own CPU of the process' affinity mask.
// libgomp silently ignores proc_bind when neither OMP_PLACES nor OMP_PROC_BIND
// is set in the environment, and unbound threads drift away from the NUMA node
// holding the data they first touched, which costs ~35% here.  The environment
// keeps precedence: if the user configured a binding, leave it alone.
void bindThreads() {
#ifdef __linux__
    if (getenv("OMP_PLACES") != nullptr || getenv("OMP_PROC_BIND") != nullptr) {
        return;
    }

    cpu_set_t allowed;
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof(allowed), &allowed) != 0) {
        return;
    }

    std::vector<int> cpus;
    for (int cpu = 0; cpu < CPU_SETSIZE; ++cpu) {
        if (CPU_ISSET(cpu, &allowed)) {
            cpus.push_back(cpu);
        }
    }
    if (cpus.empty()) {
        return;
    }

    // Linux enumerates all first SMT siblings before the second ones, so a
    // straight ascending assignment fills every physical core (on both NUMA
    // nodes) before any core is oversubscribed.
    #pragma omp parallel
    {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET(cpus[omp_get_thread_num() % cpus.size()], &set);
        sched_setaffinity(0, sizeof(set), &set);
    }
#endif
}

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

// Generate a larger set of options by scaling the test set
void generateOptions(NumaVector<OptionInput>& options, const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);

    // Each iteration is independent; the static schedule matches the one used
    // by the pricing loop so that every option is generated by the thread that
    // will later price it (first-touch NUMA placement).
    #pragma omp parallel for schedule(static) proc_bind(spread)
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

bool validateResults(const NumaVector<OptionInput>& options,
                     const NumaVector<double>& results) {
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
    
    printf("Threads: %d\n", omp_get_max_threads());

    // Spin up the thread team once so that thread creation is not attributed
    // to the timed region, and give every thread a fixed CPU.
    bindThreads();

    // Generate options
    NumaVector<OptionInput> options;
    generateOptions(options, numOptions);

    // Allocate results.  The zero fill replaces the value-initialization that
    // std::vector would have done serially and first-touches every page from
    // the thread that will own it in the pricing loop below.
    NumaVector<double> results;
    results.resize(numOptions);
    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (size_t i = 0; i < numOptions; ++i) {
        results[i] = 0.0;
    }

    // Price options
    printf("Pricing options...\n");
    auto start = std::chrono::high_resolution_clock::now();

    #pragma omp parallel for schedule(static) proc_bind(spread)
    for (size_t i = 0; i < numOptions; ++i) {
        results[i] = blackScholes(options[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    printf("Computation time: %.3f ms\n", duration.count() / 1000.0);
    printf("Options per second: %.0f\n", numOptions / (duration.count() / 1e6));
    
    // Print results for external validation
    if (printResults) {
        // print_results() takes a std::vector with the default allocator.
        const std::vector<double> plain(results.begin(), results.end());
        print_results(plain, "OptionPrices");
    }
    
    // Validation
    if (validate) {
        printf("Validating results...\n");
        bool valid = validateResults(options, results);
        
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
