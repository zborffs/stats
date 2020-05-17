//
// Created by Zach Bortoff on 2020-05-15.
//

#include "interpolation.hpp"

/// Standard Library Includes
#include <array>
#include <cmath>
#include <vector>
#include <tuple>
#include <chrono>
#include <random>
#include <fstream>
#include <filesystem>

/// External Library Includes
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <gsl/gsl_statistics_double.h>

/// Internal Library Includes
#include "stats.hpp"
#include "interpolation.hpp"

/// This is arbitrary, but std::numeric_limits<double>::min() is like 120 orders of magnitude off.
constexpr double DESIRED_SIG_FIG_ERROR = 10;
constexpr double ERROR = 0.5 * gcem::pow(10, 2 - DESIRED_SIG_FIG_ERROR);

/**
 * A class containing the boilerplate code for all the tests
 */
class InterpolationTester : public ::testing::Test {
protected:
    std::vector<double> vector_empty;
    std::vector<double> x1; std::vector<double> y1;
    std::vector<double> x2; std::vector<double> y2;

    const std::vector<const int> sizes{{2, 4, 8, 16, 128, 1024}};
    std::default_random_engine e;
    std::normal_distribution<double> d;

protected:
    virtual void SetUp() {
        /// Initialize some of the input vectors with standard test data
        vector_empty = std::vector<double> ({});
        x1 = std::vector<double>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10}); // evenly spaced
        y1 = std::vector<double>({2, 4, 6, 8, 10, 12, 14, 16, 18, 20}); // Pearson Coefficient of 1
        x2 = std::vector<double>({1, 3, 4, 7, 14, 20}); // Not evenly spaced
        y2 = std::vector<double>({-2, -6, -8, -14, -28, -40}); // Pearson Coefficient of -1;

        /// instantiate a random number generator and a normal distribution, itself seeded by uniform real distributions
        /// for the mean and standard deviation
        unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
        e = std::default_random_engine(seed);
        std::uniform_real_distribution<double> mean_distro(-(1 << 20), +(1 << 20));
        std::uniform_real_distribution<double> std_dev_distro(0, +(1 << 20));
        d = std::normal_distribution<double>(mean_distro(e), std_dev_distro(e));
    }
};

/**
 * Tests the Pearson Correlation Coefficient algorithm
 */
TEST_F(InterpolationTester, Pearson_Correlation_Coefficient) {
#ifndef NDEBUG
    /// make sure that std::logic_error is thrown if not in debug mode
#endif // NDEBUG
    /// pearson correlation coefficient ought to be 1.0

    /// pearson correlation coefficient ought to be -1.0
}

TEST_F(InterpolationTester, LinearInterpolation) {

}

TEST_F(InterpolationTester, QuadraticInterpolation) {

}

TEST_F(InterpolationTester, LogisticInterpolation) {

}

TEST_F(InterpolationTester, LinearRegression) {

}

TEST_F(InterpolationTester, QuadraticRegression) {

}

TEST_F(InterpolationTester, LogisticRegression) {

}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}