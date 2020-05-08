//
// Created by Zach Bortoff on 2020-03-02.
//

/// Standard Library Includes
#include <array>
#include <cmath>
#include <vector>
#include <tuple>

/// External Library Includes
#include <gtest/gtest.h>

/// Internal Library Includes
#include "stats.hpp"
#include "bootstrap.hpp"

/// This is arbitrary, but std::numeric_limits<double>::min() is like 120 orders of magnitude off.
const double ERROR = std::pow(10, -10);

class BootstrapTester : public ::testing::Test {
protected:
    std::vector<double> vector_empty;
    std::vector<int> coin_flip; // 1 = tails, 0 = heads

protected:
    virtual void SetUp() {
        vector_empty = std::vector<double> ({});
        std::vector<int> heads(97 + 108 + 109, 1);
        std::vector<int> tails(103 + 92 + 91, 0);
        coin_flip = std::vector<int>(heads.begin(), heads.end());
        coin_flip.reserve(400);
        for (int tail : tails) {
            coin_flip.push_back(tail);
        }
    }
};

TEST_F(BootstrapTester, ComputeBootStrapEst) {
#ifdef NDEBUG
//    ASSERT_THROW(stats::compute_bootstrap_conf_int(vector_empty.begin(), vector_empty.end()), std::logic_error);
//    ASSERT_THROW(stats::compute_bootstrap_conf_int(day1.begin(), day1.begin()), std::logic_error);
#endif // NDEBUG
    using itr = std::vector<int>::iterator;
    using Fn = std::function<double(std::vector<int>::iterator, std::vector<int>::iterator)>;
    std::vector<Fn> funcs;
    funcs.reserve(3);
    Fn f = stats::mean<itr >;
    funcs.push_back(f);
    f = stats::median<itr>;
    funcs.push_back(f);
    f = stats::var<itr>;
    funcs.push_back(f);
    stats::Bootstrap<itr, double> bootstrap(coin_flip.begin(), coin_flip.end(), funcs);
    bootstrap.monte_carlo(1000);
//    auto ci = bootstrap.confidence_interval();
//
//    std::cout << "Mean: [" << std::get<0>(ci[0]) << "-" << std::get<1>(ci[0]) << std::endl;
//    std::cout << "Median: [" << std::get<0>(ci[1]) << "-" << std::get<1>(ci[1]) << std::endl;
//    std::cout << "Variation: [" << std::get<0>(ci[2]) << "-" << std::get<1>(ci[2]) << std::endl;
}
int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}