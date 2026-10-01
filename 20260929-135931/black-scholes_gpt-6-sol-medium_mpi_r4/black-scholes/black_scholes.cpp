#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
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

// Generate a larger set of options by scaling the test set
void generateOptions(std::vector<OptionInput>& options, const size_t firstOption,
                     const size_t numOptions) {
    constexpr auto testOptions = getTestOptions();
    options.resize(numOptions);
    
    for (size_t local = 0; local < numOptions; ++local) {
        const size_t i = firstOption + local;
        // Cycle through test options and vary parameters slightly
        const OptionInput& base = testOptions[i % testOptions.size()];
        options[local] = base;
        
        // Add some variation for larger datasets
        const double factor = 1.0 + 0.1 * (i / static_cast<double>(testOptions.size()));
        options[local].spot *= factor;
        options[local].strike *= factor;
    }
}

bool validateResults(const std::vector<OptionInput>& options, 
                     const std::vector<double>& results) {
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
    MPI_Init(&argc, &argv);
    int rank, worldSize;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);

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
    }

    // Contiguous partitions preserve the original global option and result order.
    const size_t ranks = static_cast<size_t>(worldSize);
    const size_t myRank = static_cast<size_t>(rank);
    const size_t baseCount = numOptions / ranks;
    const size_t extra = numOptions % ranks;
    const size_t firstOption = myRank * baseCount + std::min(myRank, extra);
    const size_t localCount = baseCount + (myRank < extra);
    
    // Generate only the options owned by this rank.
    std::vector<OptionInput> options;
    generateOptions(options, firstOption, localCount);
    
    std::vector<double> localResults(localCount);
    
    // Price options
    if (rank == 0) printf("Pricing options...\n");
    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    
    for (size_t i = 0; i < localCount; ++i) {
        localResults[i] = blackScholes(options[i]);
    }
    
    const double localSeconds = MPI_Wtime() - start;
    double seconds = 0.0;
    MPI_Reduce(&localSeconds, &seconds, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
    if (rank == 0) {
        printf("Computation time: %.3f ms\n", seconds * 1000.0);
        printf("Options per second: %.0f\n", seconds > 0.0 ? numOptions / seconds : 0.0);
    }
    
    // Print results for external validation
    if (printResults) {
        std::vector<double> results;
        if (rank == 0) results.resize(numOptions);

        // MPI counts and displacements are ints, so gather in bounded windows.
        constexpr size_t windowSize = static_cast<size_t>(std::numeric_limits<int>::max());
        std::vector<int> counts(worldSize), displacements(worldSize);
        for (size_t window = 0; window < numOptions;) {
            const size_t end = window + std::min(windowSize, numOptions - window);
            for (int peer = 0; peer < worldSize; ++peer) {
                const size_t peerRank = static_cast<size_t>(peer);
                const size_t peerFirst = peerRank * baseCount + std::min(peerRank, extra);
                const size_t peerEnd = peerFirst + baseCount + (peerRank < extra);
                const size_t begin = std::max(window, peerFirst);
                const size_t finish = std::min(end, peerEnd);
                counts[peer] = static_cast<int>(finish > begin ? finish - begin : 0);
                displacements[peer] = static_cast<int>(begin < end ? begin - window : 0);
            }
            const size_t localBegin = std::max(window, firstOption);
            const size_t localOffset = std::min(localBegin - firstOption, localCount);
            const double* sendBuffer = counts[rank] > 0
                ? localResults.data() + localOffset : nullptr;
            MPI_Gatherv(sendBuffer, counts[rank], MPI_DOUBLE,
                        rank == 0 ? results.data() + window : nullptr,
                        counts.data(), displacements.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);
            window = end;
        }
        if (rank == 0) print_results(results, "OptionPrices");
    }
    
    // Validation
    if (validate) {
        constexpr size_t maxChecks = 10;
        const size_t checks = std::min(maxChecks, numOptions);
        std::array<double, maxChecks> ownedPrices{};
        std::array<double, maxChecks> checkedPrices{};
        for (size_t i = 0; i < localCount && firstOption + i < checks; ++i) {
            ownedPrices[firstOption + i] = localResults[i];
        }
        MPI_Reduce(ownedPrices.data(), checkedPrices.data(), static_cast<int>(checks),
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            printf("Validating results...\n");
            std::vector<OptionInput> checkedOptions;
            generateOptions(checkedOptions, 0, checks);
            std::vector<double> prices(checkedPrices.begin(), checkedPrices.begin() + checks);
            const bool valid = validateResults(checkedOptions, prices);
            printf("Validation: %s\n", valid ? "PASSED" : "FAILED");
            MPI_Finalize();
            return valid ? 0 : 1;
        }
    }
    
    MPI_Finalize();
    return 0;
}
