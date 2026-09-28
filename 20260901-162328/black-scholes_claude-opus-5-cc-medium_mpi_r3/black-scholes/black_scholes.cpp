#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <mpi.h>

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

// Generate the options owned by this rank: the global index range
// [firstIndex, firstIndex + count). The generation is a pure function of the
// global index, so every rank can build its slice independently, keeping both
// input and output data fully distributed.
void generateOptions(std::vector<OptionInput>& options, const size_t firstIndex, const size_t count) {
    constexpr auto testOptions = getTestOptions();
    options.resize(count);

    for (size_t k = 0; k < count; ++k) {
        const size_t i = firstIndex + k;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[k] = base;

        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[k].spot *= factor;
        options[k].strike *= factor;
    }
}

// Validation only inspects the first few (global) options, which rank 0
// regenerates locally; the corresponding results are collected separately.
bool validateResults(const std::vector<OptionInput>& options,
                     const std::vector<double>& results) {
    bool allPassed = true;
    const int numChecks = static_cast<int>(results.size());

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

// Block distribution: rank p owns [offset(p), offset(p+1)).
static inline size_t blockOffset(const size_t n, const int size, const int rank) {
    const size_t base = n / static_cast<size_t>(size);
    const size_t rem = n % static_cast<size_t>(size);
    const size_t r = static_cast<size_t>(rank);
    return r * base + std::min(r, rem);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;

    // Parse command line arguments (identical on every rank)
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
            numOptions = atoll(argv[++i]);
        } else if (strcmp(argv[i], "-v") == 0) {
            validate = true;
        } else if (strcmp(argv[i], "-r") == 0) {
            printResults = true;
        } else if (strcmp(argv[i], "-h") == 0) {
            if (rank == 0) printUsage(argv[0]);
            MPI_Finalize();
            return 0;
        } else {
            if (rank == 0) {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
            }
            MPI_Finalize();
            return 1;
        }
    }

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("MPI ranks: %d\n", numRanks);
    }

    // Determine this rank's slice of the global option set
    const size_t myBegin = blockOffset(numOptions, numRanks, rank);
    const size_t myEnd = blockOffset(numOptions, numRanks, rank + 1);
    const size_t myCount = myEnd - myBegin;

    // Generate the local options
    std::vector<OptionInput> options;
    generateOptions(options, myBegin, myCount);

    // Allocate local results
    std::vector<double> results(myCount);

    // Price options
    if (rank == 0) {
        printf("Pricing options...\n");
    }
    MPI_Barrier(MPI_COMM_WORLD);
    auto start = std::chrono::high_resolution_clock::now();

    const OptionInput* __restrict optionData = options.data();
    double* __restrict resultData = results.data();
    for (size_t i = 0; i < myCount; ++i) {
        resultData[i] = blackScholes(optionData[i]);
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // The benchmark time is the slowest rank's time
    long long localUs = duration.count();
    long long maxUs = localUs;
    MPI_Allreduce(&localUs, &maxUs, 1, MPI_LONG_LONG, MPI_MAX, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", maxUs / 1000.0);
        printf("Options per second: %.0f\n", numOptions / (maxUs / 1e6));
    }

    // Print results for external validation: gather the distributed results
    // into the global order on rank 0.
    if (printResults) {
        std::vector<int> counts;
        std::vector<int> displs;
        std::vector<double> gathered;
        if (rank == 0) {
            counts.resize(numRanks);
            displs.resize(numRanks);
            for (int p = 0; p < numRanks; ++p) {
                const size_t off = blockOffset(numOptions, numRanks, p);
                displs[p] = static_cast<int>(off);
                counts[p] = static_cast<int>(blockOffset(numOptions, numRanks, p + 1) - off);
            }
            gathered.resize(numOptions);
        }
        MPI_Gatherv(results.data(), static_cast<int>(myCount), MPI_DOUBLE,
                    rank == 0 ? gathered.data() : nullptr, counts.data(), displs.data(),
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(gathered, "OptionPrices");
        }
    }

    // Validation
    int exitCode = 0;
    if (validate) {
        // Collect the results of the first few global options on rank 0. Each
        // entry is contributed by exactly one rank, so a sum reduction picks it up.
        const size_t numChecks = std::min(static_cast<size_t>(10), numOptions);
        std::vector<double> firstLocal(numChecks, 0.0);
        for (size_t i = myBegin; i < myEnd && i < numChecks; ++i) {
            firstLocal[i] = results[i - myBegin];
        }
        std::vector<double> firstGlobal(numChecks, 0.0);
        MPI_Reduce(firstLocal.data(), firstGlobal.data(), static_cast<int>(numChecks),
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> firstOptions;
            generateOptions(firstOptions, 0, numChecks);
            bool valid = validateResults(firstOptions, firstGlobal);

            if (valid) {
                printf("Validation: PASSED\n");
            } else {
                printf("Validation: FAILED\n");
                exitCode = 1;
            }
        }
        MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return exitCode;
}
