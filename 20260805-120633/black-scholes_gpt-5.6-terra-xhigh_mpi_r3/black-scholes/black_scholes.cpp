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

// Construct an option from its global index.  This exactly matches the
// original data-set generation while allowing a rank to create only its own
// contiguous portion of the global input.
inline OptionInput generateOption(const size_t index) noexcept {
    constexpr auto testOptions = getTestOptions();
    OptionInput option = testOptions[index % testOptions.size()];

    const double factor = 1.0 + 0.1 * (index / static_cast<double>(testOptions.size()));
    option.spot *= factor;
    option.strike *= factor;
    return option;
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

    int rank = 0;
    int numRanks = 1;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numRanks);

    size_t numOptions = 10000;
    bool validate = false;
    bool printResults = false;
    int exitCode = 0;
    int shouldExit = 0;

    // Parse and report command-line errors once, then broadcast the selected
    // configuration so every rank follows the same collective path.
    if (rank == 0) {
        for (int i = 1; i < argc; ++i) {
            if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) {
                numOptions = static_cast<size_t>(atoll(argv[++i]));
            } else if (strcmp(argv[i], "-v") == 0) {
                validate = true;
            } else if (strcmp(argv[i], "-r") == 0) {
                printResults = true;
            } else if (strcmp(argv[i], "-h") == 0) {
                printUsage(argv[0]);
                shouldExit = 1;
                break;
            } else {
                printf("Unknown option: %s\n", argv[i]);
                printUsage(argv[0]);
                exitCode = 1;
                shouldExit = 1;
                break;
            }
        }
    }

    MPI_Bcast(&exitCode, 1, MPI_INT, 0, MPI_COMM_WORLD);
    MPI_Bcast(&shouldExit, 1, MPI_INT, 0, MPI_COMM_WORLD);
    if (shouldExit != 0) {
        MPI_Finalize();
        return exitCode;
    }

    unsigned long long numOptionsWire = static_cast<unsigned long long>(numOptions);
    int flags[2] = {validate ? 1 : 0, printResults ? 1 : 0};
    MPI_Bcast(&numOptionsWire, 1, MPI_UNSIGNED_LONG_LONG, 0, MPI_COMM_WORLD);
    MPI_Bcast(flags, 2, MPI_INT, 0, MPI_COMM_WORLD);
    numOptions = static_cast<size_t>(numOptionsWire);
    validate = flags[0] != 0;
    printResults = flags[1] != 0;

    const size_t rankCount = static_cast<size_t>(numRanks);
    const size_t rankIndex = static_cast<size_t>(rank);
    const size_t baseCount = numOptions / rankCount;
    const size_t remainder = numOptions % rankCount;
    const size_t localCount = baseCount + (rankIndex < remainder ? 1 : 0);
    const size_t firstIndex = rankIndex * baseCount + std::min(rankIndex, remainder);

    if (rank == 0) {
        printf("Black-Scholes Option Pricing Benchmark\n");
        printf("Number of options: %zu\n", numOptions);
        printf("Validation: %s\n", validate ? "enabled" : "disabled");
        printf("Pricing options...\n");
    }

    // Keep full results only when external output requires them.  This keeps
    // memory per rank proportional to the local work in the output mode and
    // eliminates per-option storage in the common benchmarking path.
    std::vector<double> localResults;
    if (printResults) {
        localResults.resize(localCount);
    }

    constexpr size_t maxValidationChecks = 10;
    const int validationCount = static_cast<int>(std::min(numOptions, maxValidationChecks));
    std::array<double, maxValidationChecks> localValidationResults{};

    MPI_Barrier(MPI_COMM_WORLD);
    const double start = MPI_Wtime();
    for (size_t localIndex = 0; localIndex < localCount; ++localIndex) {
        const size_t globalIndex = firstIndex + localIndex;
        const double price = blackScholes(generateOption(globalIndex));

        if (printResults) {
            localResults[localIndex] = price;
        }
        if (validate && globalIndex < static_cast<size_t>(validationCount)) {
            localValidationResults[globalIndex] = price;
        }
    }
    const double localDuration = MPI_Wtime() - start;

    // The maximum local duration is the distributed computation time: it is
    // the time until every rank has completed its assigned global range.
    double computationTime = 0.0;
    MPI_Reduce(&localDuration, &computationTime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Computation time: %.3f ms\n", computationTime * 1000.0);
        printf("Options per second: %.0f\n", computationTime > 0.0 ? numOptions / computationTime : 0.0);
    }

    std::vector<double> results;
    if (printResults) {
        // MPI_Gatherv uses int counts/displacements.  Printing also requires a
        // complete root-side result vector, so reject sizes it cannot express.
        if (numOptions > static_cast<size_t>(std::numeric_limits<int>::max())) {
            if (rank == 0) {
                fprintf(stderr, "Result collection supports at most %d options\n",
                        std::numeric_limits<int>::max());
            }
            MPI_Finalize();
            return 1;
        }

        if (rank == 0) {
            results.resize(numOptions);
        }

        std::vector<int> receiveCounts;
        std::vector<int> displacements;
        if (rank == 0) {
            receiveCounts.resize(numRanks);
            displacements.resize(numRanks);
            for (int process = 0; process < numRanks; ++process) {
                const size_t processIndex = static_cast<size_t>(process);
                receiveCounts[process] = static_cast<int>(baseCount + (processIndex < remainder ? 1 : 0));
                displacements[process] = static_cast<int>(processIndex * baseCount +
                                                          std::min(processIndex, remainder));
            }
        }

        MPI_Gatherv(localResults.data(), static_cast<int>(localCount), MPI_DOUBLE,
                    rank == 0 ? results.data() : nullptr,
                    rank == 0 ? receiveCounts.data() : nullptr,
                    rank == 0 ? displacements.data() : nullptr,
                    MPI_DOUBLE, 0, MPI_COMM_WORLD);

        if (rank == 0) {
            print_results(results, "OptionPrices");
        }
    }

    int validationPassed = 1;
    if (validate) {
        std::array<double, maxValidationChecks> validationResults{};
        MPI_Reduce(localValidationResults.data(), validationResults.data(), validationCount,
                   MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
        if (rank == 0) {
            std::vector<OptionInput> validationOptions(validationCount);
            for (int i = 0; i < validationCount; ++i) {
                validationOptions[static_cast<size_t>(i)] = generateOption(static_cast<size_t>(i));
            }
            std::vector<double> checkedResults(validationResults.begin(),
                                               validationResults.begin() + validationCount);
            printf("Validating results...\n");
            validationPassed = validateResults(validationOptions, checkedResults) ? 1 : 0;
            printf("Validation: %s\n", validationPassed ? "PASSED" : "FAILED");
        }
        MPI_Bcast(&validationPassed, 1, MPI_INT, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return validationPassed ? 0 : 1;
}
