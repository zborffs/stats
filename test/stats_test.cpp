//
// Created by Zach Bortoff on 2019-10-12.
//

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

/// This is arbitrary, but std::numeric_limits<double>::min() is like 120 orders of magnitude off.
constexpr double DESIRED_SIG_FIG_ERROR = 10;
constexpr double ERROR = 0.5 * gcem::pow(10, 2 - DESIRED_SIG_FIG_ERROR);

std::vector<std::string> split(const std::string& str, const char delim) {
    std::vector<std::string> ret;
    std::string::const_iterator first  = str.cbegin();
    std::string::const_iterator second = str.cbegin();

    for(; second <= str.end(); ++second) {
        if(*(second) == delim || second == str.end()) {
            if(second != first) {
                ret.emplace_back(first, second);
            }

            first = second + 1;
        }
    }

    return ret;
}
std::vector<std::string> read_file(const std::string& file_name) {
    std::vector<std::string> lines;

    try {
        std::fstream in(file_name);
        in.exceptions(std::ifstream::badbit);
        if(!in.is_open()) {
            std::string err("Failed to open file: ");
            err += file_name;
            throw std::runtime_error(err);
        }

        std::string line;
        for(int i = 1; std::getline(in, line); i++) {
            lines.push_back(line);
        }

        if(in.bad()) {
            throw std::runtime_error("Runtime error in read_fen_from_file(const std::string&, const uint32): Badbit file.");
        }

        in.close();

    } catch(const std::exception& e) {
        throw;
    }

    return lines;
}

class StatsTester : public ::testing::Test {
protected:
    std::vector<double> vector_empty;
    std::vector<double> vector_double_odd_sorted;
    std::vector<int> vector_int_even_sorted;
    std::vector<double> vector_double_odd_repeat_unsorted;
    std::vector<float> vector_float_odd_repeat_sorted;

    const std::vector<const int> sizes{{2, 4, 8, 16, 128, 1024}};
    std::default_random_engine e;
    std::normal_distribution<double> d;

protected:
    virtual void SetUp() {
        vector_empty = std::vector<double> ({});
        vector_double_odd_sorted = std::vector<double>({1.0, 2.0, 3.0, 4.0, 5.0});
        vector_int_even_sorted = std::vector<int> ({3, 5, 7, 9});
        vector_double_odd_repeat_unsorted = std::vector<double>({5, 1, 6, 6});
        vector_float_odd_repeat_sorted = std::vector<float>({-1, 0, 0, 0, 4, 4, 0});

        unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
        e = std::default_random_engine(seed);
        std::uniform_real_distribution<double> mean_distro(-(1 << 20), +(1 << 20));
        std::uniform_real_distribution<double> std_dev_distro(0, +(1 << 20));
        d = std::normal_distribution<double>(mean_distro(e), std_dev_distro(e));
    }
};

TEST_F(StatsTester, Absolute) {
    double d1 = 1.0, d2 = -0.123, d3 = 0.0, d4 = -0.0;
    unsigned long int ul1 = 1, ul2 = 0;
    int i1 = -12, i2 = 1, i3 = 0;
    long long ll1 = -123, ll2 = 39592041, ll3 = 0;
    float f1 = 0.12345, f2 = -0.321, f3 = -0.0, f4 = 0.0;
    long l1 = -22, l2 = 45, l3 = 0;
    unsigned int u1 = 33, u2 = 0;

    EXPECT_FLOAT_EQ(st::internal::abs(d1), 1.0);
    EXPECT_FLOAT_EQ(st::internal::abs(d2), 0.123);
    EXPECT_FLOAT_EQ(st::internal::abs(d3), 0.0);
    EXPECT_FLOAT_EQ(st::internal::abs(d4), 0.0);

    EXPECT_EQ(st::internal::abs(ul1), 1);
    EXPECT_EQ(st::internal::abs(ul2), 0);

    EXPECT_EQ(st::internal::abs(i1), 12);
    EXPECT_EQ(st::internal::abs(i2), 1);
    EXPECT_EQ(st::internal::abs(i3), 0);
    EXPECT_EQ(st::internal::abs(ll1), 123);
    EXPECT_EQ(st::internal::abs(ll2), ll2);
    EXPECT_EQ(st::internal::abs(ll3), 0);
    EXPECT_FLOAT_EQ(st::internal::abs(f1), f1);
    EXPECT_FLOAT_EQ(st::internal::abs(f2), 0.321);
    EXPECT_FLOAT_EQ(st::internal::abs(f3), 0.0);
    EXPECT_FLOAT_EQ(st::internal::abs(f4), 0.0);
    EXPECT_EQ(st::internal::abs(l1), 22);
    EXPECT_EQ(st::internal::abs(l2), 45);
    EXPECT_EQ(st::internal::abs(l3), 0);
    EXPECT_EQ(st::internal::abs(u1), 33);
    EXPECT_EQ(st::internal::abs(u2), 0);
}

TEST_F(StatsTester, Error) {
    double d_m1 = 10.05;
    double d_m2 = 9.95;
    double d_a = 10.0;

    unsigned int ui_m1 = 1;
    unsigned int ui_m2 = 3;
    unsigned int ui_a = 3;

    long long ll_a = 123;
    long long ll_b = -133;

    /// Absolute Error
    EXPECT_NEAR(st::abs_err(d_m1, d_a), 0.05, ERROR);
    EXPECT_NEAR(st::abs_err(d_m2, d_a), 0.05, ERROR);
    EXPECT_EQ(st::abs_err(ui_m1, ui_a), 2);
    EXPECT_EQ(st::abs_err(ui_m2, ui_a), 0);
    EXPECT_EQ(st::abs_err(ll_a, ll_b), 256);

    /// Relative Error
    EXPECT_NEAR(st::rel_err(d_m1, d_a), 0.005, ERROR);
    EXPECT_NEAR(st::rel_err(d_m2, d_a), 0.005, ERROR);
    EXPECT_NEAR(st::rel_err(ui_m1, ui_a), 2.0 / 3.0, ERROR);
    EXPECT_EQ(st::rel_err(ui_m2, ui_a), 0.0);
    EXPECT_NEAR(st::rel_err(ll_a, ll_b), (256.0 / 133.0), ERROR);

    /// Percent Error
    EXPECT_NEAR(st::perc_err(d_m1, d_a), 0.5, ERROR);
    EXPECT_NEAR(st::perc_err(d_m2, d_a), 0.5, ERROR);
    EXPECT_NEAR(st::perc_err(ui_m1, ui_a), 200.0 / 3.0, ERROR);
    EXPECT_EQ(st::perc_err(ui_m2, ui_a), 0.0);
    EXPECT_NEAR(st::perc_err(ll_a, ll_b), 256.0 / 1.33, ERROR);
}

TEST_F(StatsTester, Maximum) {
#ifdef NDEBUG
    ASSERT_THROW(st::max(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::max(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);
        Eigen::RowVectorXd eigen_row_data(size);
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
            eigen_row_data(j) = datum;
        }
        Eigen::VectorXd eigen_data = eigen_row_data.transpose();

        double gsl_ans = gsl_stats_max(gsl_data, 1, size);
        double stl_ans = st::max(stl_data.begin(), stl_data.end());
        double eigen_ans = st::max(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }

}

TEST_F(StatsTester, Minimum) {
#ifdef NDEBUG
    ASSERT_THROW(st::min(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::min(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);
        Eigen::RowVectorXd eigen_row_data(size);
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
            eigen_row_data(j) = datum;
        }
        Eigen::VectorXd eigen_data = eigen_row_data.transpose();

        double gsl_ans = gsl_stats_min(gsl_data, 1, size);
        double stl_ans = st::min(stl_data.begin(), stl_data.end());
        double eigen_ans = st::min(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }
}

TEST_F(StatsTester, Median) {
#ifdef NDEBUG
    ASSERT_THROW(st::median(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::median(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);
        Eigen::RowVectorXd eigen_row_data(size);
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
            eigen_row_data(j) = datum;
        }
        Eigen::VectorXd eigen_data = eigen_row_data.transpose();

        double gsl_ans = gsl_stats_median(gsl_data, 1, size);
        double stl_ans = st::median(stl_data.begin(), stl_data.end());
        double eigen_ans = st::median(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }
}

TEST_F(StatsTester, Mode) {
#ifdef NDEBUG
    ASSERT_THROW(st::mode(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::mode(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_EQ(st::mode(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), std::unordered_set<double>({1.0, 2.0, 3.0, 4.0, 5.0}));
    EXPECT_EQ(st::mode(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), std::unordered_set<int>({3, 5, 7, 9}));
    EXPECT_EQ(st::mode(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), std::unordered_set<double>({6}));
    EXPECT_EQ(st::mode(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), std::unordered_set<float>({0.0}));
}

TEST_F(StatsTester, ArithmeticMean) {
#ifdef NDEBUG
    ASSERT_THROW(st::mean(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::mean(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);
        Eigen::RowVectorXd eigen_row_data(size);
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
            eigen_row_data(j) = datum;
        }
        Eigen::VectorXd eigen_data = eigen_row_data.transpose();

        double gsl_ans = gsl_stats_sd(gsl_data, 1, size);
        double stl_ans = st::std_dev(stl_data.begin(), stl_data.end());
        double eigen_ans = st::std_dev(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }
}

TEST_F(StatsTester, Range) {
#ifdef NDEBUG
    ASSERT_THROW(st::range(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::range(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_NEAR(st::range(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), 4.0, ERROR);
    EXPECT_NEAR(st::range(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), 6.0, ERROR);
    EXPECT_NEAR(st::range(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), 5.0, ERROR);
    EXPECT_NEAR(st::range(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), 5.0, ERROR);
}

TEST_F(StatsTester, Quartiles) {
#ifdef NDEBUG
    ASSERT_THROW(st::quartiles(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::quartiles(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);

        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
        }

        std::sort(&gsl_data[0], &gsl_data[size]);
        std::sort(stl_data.begin(), stl_data.end());

        double q1 = gsl_stats_quantile_from_sorted_data(gsl_data, 1, size, 0.25);
        double q2 = gsl_stats_quantile_from_sorted_data(gsl_data, 1, size, 0.50);
        double q3 = gsl_stats_quantile_from_sorted_data(gsl_data, 1, size, 0.75);
        auto [q1_1, q2_1, q3_1] = st::quartiles(stl_data.begin(), stl_data.end());
        EXPECT_NEAR(q1, q1_1, ERROR);
        EXPECT_NEAR(q2, q2_1, ERROR);
        EXPECT_NEAR(q3, q3_1, ERROR);
    }
}

TEST_F(StatsTester, InterquartileRange) {
#ifdef NDEBUG
    ASSERT_THROW(st::interquartile_range(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::interquartile_range(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);

        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
        }

        std::sort(&gsl_data[0], &gsl_data[size]);
        std::sort(stl_data.begin(), stl_data.end());

        double q1 = gsl_stats_quantile_from_sorted_data(gsl_data, 1, size, 0.25);
        double q3 = gsl_stats_quantile_from_sorted_data(gsl_data, 1, size, 0.75);
        double iqr = st::interquartile_range(stl_data.begin(), stl_data.end());
        EXPECT_NEAR(q3 - q1, iqr, ERROR);
    }
}

TEST_F(StatsTester, Outliers) {
#ifdef NDEBUG
    ASSERT_THROW(st::outliers(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::outliers(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG
    std::array<long, 7> outlier_array({28, 26, 29, 30, 81, 32, 37});
    std::vector<unsigned int> outlier_vec({16, 14, 3, 12, 15, 17, 22, 15, 52});

    EXPECT_EQ(st::outliers(outlier_array.begin(), outlier_array.end()), std::unordered_set<long>{});
    EXPECT_EQ(st::outliers(outlier_vec.begin(), outlier_vec.end()), std::unordered_set<unsigned int>({52}));
}

TEST_F(StatsTester, MedianAbsoluteDeviation) {
#ifdef NDEBUG
    ASSERT_THROW(st::median_abs_dev(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::median_abs_dev(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_NEAR(st::median_abs_dev(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), 1.0, ERROR);
    EXPECT_NEAR(st::median_abs_dev(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), 2.0, ERROR);
    EXPECT_NEAR(st::median_abs_dev(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), 0.5, ERROR);
    EXPECT_NEAR(st::median_abs_dev(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), 0.0, ERROR);
}

/// add (..., mean) test
TEST_F(StatsTester, StandardDeviation) {
#ifdef NDEBUG
    ASSERT_THROW(st::std_dev(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::std_dev(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);
        Eigen::RowVectorXd eigen_row_data(size);
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
            eigen_row_data(j) = datum;
        }
        Eigen::VectorXd eigen_data = eigen_row_data.transpose();

        double gsl_ans = gsl_stats_sd(gsl_data, 1, size);
        double stl_ans = st::std_dev(stl_data.begin(), stl_data.end());
        double eigen_ans = st::std_dev(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }
}

/// add (..., mean) test
TEST_F(StatsTester, Variation) {
#ifdef NDEBUG
    ASSERT_THROW(st::var(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::var(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG
    const double ERR = 0.001;
    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);
        Eigen::RowVectorXd eigen_row_data(size);
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
            eigen_row_data(j) = datum;
        }
        Eigen::VectorXd eigen_data = eigen_row_data.transpose();

        double gsl_ans = gsl_stats_variance(gsl_data, 1, size);
        double stl_ans = st::var(stl_data.begin(), stl_data.end());
        double eigen_ans = st::var(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERR);
    }
}

TEST_F(StatsTester, Skewness) {
#ifdef NDEBUG
    ASSERT_THROW(st::skewness(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::skewness(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG
    const double ERR = 0.001;
    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
        }

        double gsl_ans = gsl_stats_skew(gsl_data, 1, size);
        auto stl_ans = st::skewness(stl_data.begin(), stl_data.end());
        EXPECT_NEAR(stl_ans, gsl_ans, ERR);
    }
}

TEST_F(StatsTester, ExKurtosis) {
#ifdef NDEBUG
    ASSERT_THROW(st::ex_kurtosis(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::ex_kurtosis(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG
    const double ERR = 0.001;
    for (int size : sizes) {
        double gsl_data[size];
        std::vector<double> stl_data(size);
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data[j] = datum;
            stl_data[j] = datum;
        }

        double gsl_ans = gsl_stats_kurtosis(gsl_data, 1, size);
        auto stl_ans = st::ex_kurtosis(stl_data.begin(), stl_data.end());
        EXPECT_NEAR(stl_ans, gsl_ans, ERR);
    }
}

TEST_F(StatsTester, Repeatability) {
#ifdef NDEBUG
    ASSERT_THROW(st::ex_kurtosis(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::ex_kurtosis(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    /// Example data taken from https://www.youtube.com/watch?v=ckcU4iPvhlg
    std::vector<double> day1({9.66, 10.85, 10.47, 10.12, 9.78});
    std::vector<double> day2({9.09, 9.11, 9.25, 9.31});
    std::vector<double> day3({9.21, 9.24, 9.14, 9.11, 9.20});
    std::vector<double> day4({11.28, 11.12, 11.58, 10.97, 11.51});
    std::vector<double> day5({9.07, 10.04, 9.69, 10.94});
    std::vector<std::vector<double>> data({day1, day2, day3, day4, day5});

    EXPECT_NEAR(st::repeatability(data.begin(), data.end()), 0.04, 0.01);
}

TEST_F(StatsTester, InverseNormalCDF) {
    double p1 = 0.80;
    double p2 = 0.90;
    double p3 = 0.95;
    double p4 = 0.98;
    double p5 = 0.99;

    EXPECT_NEAR(st::inv_normal_cdf(p1), 1.281551565545, ERROR);
    EXPECT_NEAR(st::inv_normal_cdf(p2), 1.644853626951, ERROR);
    EXPECT_NEAR(st::inv_normal_cdf(p3), 1.959963984540, ERROR);
    EXPECT_NEAR(st::inv_normal_cdf(p4), 2.326347874041, ERROR);
    EXPECT_NEAR(st::inv_normal_cdf(p5), 2.575829303549, ERROR);
}

TEST_F(StatsTester, InterquartileMean) {
    std::vector<double> wiki_div_4_example({1, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 38});
    EXPECT_NEAR(st::interquartile_mean(wiki_div_4_example.begin(), wiki_div_4_example.end()), 6.5, ERROR);
    std::vector<double> wiki_not_div_4_example({1, 3, 5, 7, 9, 11, 13, 15, 17});
    EXPECT_NEAR(st::interquartile_mean(wiki_not_div_4_example.begin(), wiki_not_div_4_example.end()), 9, ERROR);
}

TEST_F(StatsTester, HarmonicMean) {
#ifdef NDEBUG
    /// If NBEBUG (not debug) is defined, implying that we are NOT debugging right now, then test that erroneous inputs throw exceptions
    ASSERT_THROW(st::ex_kurtosis(vector_empty.begin(), vector_empty.end()), std::logic_error);
#endif // NDEBUG

    std::vector<double> v1({1.0, 2.0, 3.0, 4.0, 5.0}); double v1_hmean = (1. + 1. / 2 + 1. / 3 + 1. / 4 + 1. / 5) / 5.;

    EXPECT_NEAR(st::hmean(v1.begin(), v1.end()), v1_hmean, ERROR);
}

TEST_F(StatsTester, TStatstic) {
    /// NIST Example
    auto current_path = std::filesystem::current_path();
    std::string t_stat_test_file = current_path.string() + "/../../test/res/t_test.dat";
    std::vector<std::string> lines;

    try {
        lines = read_file(t_stat_test_file);
    } catch (...) {
        std::cout << "Exiting!" << std::endl;
    }

    std::vector<std::string> split_line;
    std::vector<int> american_cars;
    std::vector<int> japanese_cars;

    for (auto & line : lines) {
        split_line = split(line, ' ');
        /// i happen to know there are only two items in this vector
        for (int j = 0; j < split_line.size(); j++) {
            if (j == 0) {
                american_cars.push_back(std::stoi(split_line[j]));
            } else if (j == 1) {
                japanese_cars.push_back(std::stoi(split_line[j]));
            }
        }
    }

    auto T = st::t_statistic(american_cars.begin(), american_cars.end(), japanese_cars.begin(), japanese_cars.end());
    std::cout << "T: " << T << std::endl;
    EXPECT_NEAR(T, -12.62059, 0.4);
    bool rej = st::two_samp_t_test(american_cars.begin(), american_cars.end(), japanese_cars.begin(), japanese_cars.end());
    EXPECT_EQ(rej, true);
}

/**
 * Tests the pears_corr_coeff(...) function on predefined NIST data
 */
TEST_F(StatsTester, PearsonCorrelationCoeff) {
#ifdef NDEBUG
    /// makes sure that an std::logic_error is thrown if operating on an empty vector and debugging is off
    ASSERT_THROW(st::pears_corr_coeff(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(st::pears_corr_coeff(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    /// for each element of differently sized, randomly generated input data, computes the mean using this library's
    /// implementation and compares it to the gsl library's implementation to test for correctness
    for (int size : sizes) {
        /// initializes input data
        double gsl_data_1[size];
        double gsl_data_2[size];
        std::vector<double> stl_data_1(size);
        std::vector<double> stl_data_2(size);
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data_1[j] = datum;
            stl_data_1[j] = datum;
        }
        for (int j = 0; j < size; j++) {
            double datum = d(e);
            gsl_data_2[j] = datum;
            stl_data_2[j] = datum;
        }

        /// computes the mean using the gsl function and the two st:: methods (one for STL objects, the other for
        /// Eigen::VectorXds)
        double gsl_ans = gsl_stats_correlation(gsl_data_1, 1, gsl_data_2, 1, size);
        double stl_ans = st::pears_corr_coeff(stl_data_1.begin(), stl_data_1.end(), stl_data_2.begin());
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
    }
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

