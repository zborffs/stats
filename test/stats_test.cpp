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

/// External Library Includes
#include <gtest/gtest.h>
#include <Eigen/Dense>
#include <gsl/gsl_statistics_double.h>

/// Internal Library Includes
#include "stats.hpp"

/// This is arbitrary, but std::numeric_limits<double>::min() is like 120 orders of magnitude off.
double DESIRED_SIG_FIG_ERROR = 10;
double ERROR = 0.5 * std::pow(10, 2 - DESIRED_SIG_FIG_ERROR);

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
        std::uniform_real_distribution<double> mean_distro(-(1 << 16), +(1 << 16));
        std::uniform_real_distribution<double> std_dev_distro(0, +(1 << 12));
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

    EXPECT_FLOAT_EQ(stats::internal::abs(d1), 1.0);
    EXPECT_FLOAT_EQ(stats::internal::abs(d2), 0.123);
    EXPECT_FLOAT_EQ(stats::internal::abs(d3), 0.0);
    EXPECT_FLOAT_EQ(stats::internal::abs(d4), 0.0);

    EXPECT_EQ(stats::internal::abs(ul1), 1);
    EXPECT_EQ(stats::internal::abs(ul2), 0);

    EXPECT_EQ(stats::internal::abs(i1), 12);
    EXPECT_EQ(stats::internal::abs(i2), 1);
    EXPECT_EQ(stats::internal::abs(i3), 0);
    EXPECT_EQ(stats::internal::abs(ll1), 123);
    EXPECT_EQ(stats::internal::abs(ll2), ll2);
    EXPECT_EQ(stats::internal::abs(ll3), 0);
    EXPECT_FLOAT_EQ(stats::internal::abs(f1), f1);
    EXPECT_FLOAT_EQ(stats::internal::abs(f2), 0.321);
    EXPECT_FLOAT_EQ(stats::internal::abs(f3), 0.0);
    EXPECT_FLOAT_EQ(stats::internal::abs(f4), 0.0);
    EXPECT_EQ(stats::internal::abs(l1), 22);
    EXPECT_EQ(stats::internal::abs(l2), 45);
    EXPECT_EQ(stats::internal::abs(l3), 0);
    EXPECT_EQ(stats::internal::abs(u1), 33);
    EXPECT_EQ(stats::internal::abs(u2), 0);
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
    EXPECT_NEAR(stats::abs_err(d_m1, d_a), 0.05, ERROR);
    EXPECT_NEAR(stats::abs_err(d_m2, d_a), 0.05, ERROR);
    EXPECT_EQ(stats::abs_err(ui_m1, ui_a), 2);
    EXPECT_EQ(stats::abs_err(ui_m2, ui_a), 0);
    EXPECT_EQ(stats::abs_err(ll_a, ll_b), 256);

    /// Relative Error
    EXPECT_NEAR(stats::rel_err(d_m1, d_a), 0.005, ERROR);
    EXPECT_NEAR(stats::rel_err(d_m2, d_a), 0.005, ERROR);
    EXPECT_NEAR(stats::rel_err(ui_m1, ui_a), 2.0 / 3.0, ERROR);
    EXPECT_EQ(stats::rel_err(ui_m2, ui_a), 0.0);
    EXPECT_NEAR(stats::rel_err(ll_a, ll_b), (256.0 / 133.0), ERROR);

    /// Percent Error
    EXPECT_NEAR(stats::perc_err(d_m1, d_a), 0.5, ERROR);
    EXPECT_NEAR(stats::perc_err(d_m2, d_a), 0.5, ERROR);
    EXPECT_NEAR(stats::perc_err(ui_m1, ui_a), 200.0 / 3.0, ERROR);
    EXPECT_EQ(stats::perc_err(ui_m2, ui_a), 0.0);
    EXPECT_NEAR(stats::perc_err(ll_a, ll_b), 256.0 / 1.33, ERROR);
}

TEST_F(StatsTester, Maximum) {
#ifdef NDEBUG
    ASSERT_THROW(stats::max(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::max(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
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
        double stl_ans = stats::max(stl_data.begin(), stl_data.end());
        double eigen_ans = stats::max(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }

}

TEST_F(StatsTester, Minimum) {
#ifdef NDEBUG
    ASSERT_THROW(stats::min(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::min(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
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
        double stl_ans = stats::min(stl_data.begin(), stl_data.end());
        double eigen_ans = stats::min(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }
}

TEST_F(StatsTester, Median) {
#ifdef NDEBUG
    ASSERT_THROW(stats::median(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::median(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
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
        double stl_ans = stats::median(stl_data.begin(), stl_data.end());
        double eigen_ans = stats::median(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }
}

TEST_F(StatsTester, Mode) {
#ifdef NDEBUG
    ASSERT_THROW(stats::mode(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::mode(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_EQ(stats::mode(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), std::unordered_set<double>({1.0, 2.0, 3.0, 4.0, 5.0}));
    EXPECT_EQ(stats::mode(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), std::unordered_set<int>({3, 5, 7, 9}));
    EXPECT_EQ(stats::mode(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), std::unordered_set<double>({6}));
    EXPECT_EQ(stats::mode(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), std::unordered_set<float>({0.0}));
}

TEST_F(StatsTester, ArithmeticMean) {
#ifdef NDEBUG
    ASSERT_THROW(stats::mean(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::mean(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
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
        double stl_ans = stats::std_dev(stl_data.begin(), stl_data.end());
        double eigen_ans = stats::std_dev(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }
}

TEST_F(StatsTester, Range) {
#ifdef NDEBUG
    ASSERT_THROW(stats::range(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::range(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_NEAR(stats::range(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), 4.0, ERROR);
    EXPECT_NEAR(stats::range(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), 6.0, ERROR);
    EXPECT_NEAR(stats::range(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), 5.0, ERROR);
    EXPECT_NEAR(stats::range(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), 5.0, ERROR);
}

TEST_F(StatsTester, Quartiles) {
#ifdef NDEBUG
    ASSERT_THROW(stats::quartiles(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::quartiles(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_EQ(stats::quartiles(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), std::make_tuple(1.5, 3.0, 4.5));
    EXPECT_EQ(stats::quartiles(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), std::make_tuple(4, 6, 8));
    EXPECT_EQ(stats::quartiles(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), std::make_tuple(3, 5.5, 6));
    EXPECT_EQ(stats::quartiles(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), std::make_tuple(0.0, 0.0, 4.0));
}

TEST_F(StatsTester, InterquartileRange) {
#ifdef NDEBUG
    ASSERT_THROW(stats::interquartile_range(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::interquartile_range(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG
    std::vector<int> khan_academy_example({4, 4, 10, 11, 15, 7, 14, 12, 6});
    EXPECT_NEAR(stats::interquartile_range(khan_academy_example.begin(), khan_academy_example.end()), 8.0, ERROR);
    EXPECT_NEAR(stats::interquartile_range(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), 3.0, ERROR);
    EXPECT_NEAR(stats::interquartile_range(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), 4.0, ERROR);
    EXPECT_NEAR(stats::interquartile_range(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), 3.0, ERROR);
    EXPECT_NEAR(stats::interquartile_range(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), 4.0, ERROR);
}

TEST_F(StatsTester, Outliers) {
#ifdef NDEBUG
    ASSERT_THROW(stats::outliers(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::outliers(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_EQ(stats::outliers(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), std::unordered_set<double>({}));
    EXPECT_EQ(stats::outliers(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), std::unordered_set<int>({}));
    EXPECT_EQ(stats::outliers(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), std::unordered_set<double>({}));
    EXPECT_EQ(stats::outliers(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), std::unordered_set<float>({}));


    std::array<long, 7> outlier_array({28, 26, 29, 30, 81, 32, 37});
    std::vector<unsigned int> outlier_vec({16, 14, 3, 12, 15, 17, 22, 15, 52});

    EXPECT_EQ(stats::outliers(outlier_array.begin(), outlier_array.end()), std::unordered_set<long>({81}));
    EXPECT_EQ(stats::outliers(outlier_vec.begin(), outlier_vec.end()), std::unordered_set<unsigned int>({3, 52}));
}

TEST_F(StatsTester, MedianAbsoluteDeviation) {
#ifdef NDEBUG
    ASSERT_THROW(stats::median_abs_dev(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::median_abs_dev(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_NEAR(stats::median_abs_dev(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), 1.0, ERROR);
    EXPECT_NEAR(stats::median_abs_dev(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), 2.0, ERROR);
    EXPECT_NEAR(stats::median_abs_dev(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), 0.5, ERROR);
    EXPECT_NEAR(stats::median_abs_dev(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), 0.0, ERROR);
}

TEST_F(StatsTester, StandardDeviation) {
#ifdef NDEBUG
    ASSERT_THROW(stats::std_dev(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::std_dev(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
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
        double stl_ans = stats::std_dev(stl_data.begin(), stl_data.end());
        double eigen_ans = stats::std_dev(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }
}

TEST_F(StatsTester, Variation) {
#ifdef NDEBUG
    ASSERT_THROW(stats::var(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::var(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
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

        double gsl_ans = gsl_stats_variance(gsl_data, 1, size);
        double stl_ans = stats::var(stl_data.begin(), stl_data.end());
        double eigen_ans = stats::var(eigen_data);
        EXPECT_NEAR(stl_ans, gsl_ans, ERROR);
        EXPECT_NEAR(eigen_ans, gsl_ans, ERROR);
    }
}

TEST_F(StatsTester, QuartileDeviation) {
#ifdef NDEBUG
    ASSERT_THROW(stats::quartile_dev(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::quartile_dev(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_NEAR(stats::quartile_dev(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), 1.5, ERROR);
    EXPECT_NEAR(stats::quartile_dev(vector_int_even_sorted.begin(), vector_int_even_sorted.end()), 2.0, ERROR);
    EXPECT_NEAR(stats::quartile_dev(vector_double_odd_repeat_unsorted.begin(), vector_double_odd_repeat_unsorted.end()), 1.5, ERROR);
    EXPECT_NEAR(stats::quartile_dev(vector_float_odd_repeat_sorted.begin(), vector_float_odd_repeat_sorted.end()), 2.0, ERROR);
}


TEST_F(StatsTester, ExKurtosis) {
#ifdef NDEBUG
    ASSERT_THROW(stats::ex_kurtosis(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::ex_kurtosis(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    EXPECT_NEAR(stats::ex_kurtosis(vector_double_odd_sorted.begin(), vector_double_odd_sorted.end()), -1.3, ERROR);
    std::vector<double> wiki_example({ 0, 3, 4, 1, 2, 3, 0, 2, 1, 3, 2, 0, 2, 2, 3, 2, 5, 2, 3, 999});
    EXPECT_NEAR(stats::ex_kurtosis(wiki_example.begin(), wiki_example.end()), 15.0514265438, ERROR);
}

TEST_F(StatsTester, Repeatability) {
#ifdef NDEBUG
    ASSERT_THROW(stats::ex_kurtosis(vector_empty.begin(), vector_empty.end()), std::logic_error);
    ASSERT_THROW(stats::ex_kurtosis(vector_double_odd_sorted.begin(), vector_double_odd_sorted.begin()), std::logic_error);
#endif // NDEBUG

    /// Example data taken from https://www.youtube.com/watch?v=ckcU4iPvhlg
    std::vector<double> day1({9.66, 10.85, 10.47, 10.12, 9.78});
    std::vector<double> day2({9.09, 9.11, 9.25, 9.31});
    std::vector<double> day3({9.21, 9.24, 9.14, 9.11, 9.20});
    std::vector<double> day4({11.28, 11.12, 11.58, 10.97, 11.51});
    std::vector<double> day5({9.07, 10.04, 9.69, 10.94});
    std::vector<std::vector<double>> data({day1, day2, day3, day4, day5});

    EXPECT_NEAR(stats::repeatability(data.begin(), data.end()), 0.04, 0.01);
}

TEST_F(StatsTester, InverseNormalCDF) {
    double p1 = 0.80;
    double p2 = 0.90;
    double p3 = 0.95;
    double p4 = 0.98;
    double p5 = 0.99;

    EXPECT_NEAR(stats::inv_normal_cdf(p1), 1.281551565545, ERROR);
    EXPECT_NEAR(stats::inv_normal_cdf(p2), 1.644853626951, ERROR);
    EXPECT_NEAR(stats::inv_normal_cdf(p3), 1.959963984540, ERROR);
    EXPECT_NEAR(stats::inv_normal_cdf(p4), 2.326347874041, ERROR);
    EXPECT_NEAR(stats::inv_normal_cdf(p5), 2.575829303549, ERROR);
}

TEST_F(StatsTester, InterquartileMean) {
    std::vector<double> wiki_div_4_example({1, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9, 38});
    EXPECT_NEAR(stats::interquartile_mean(wiki_div_4_example.begin(), wiki_div_4_example.end()), 6.5, ERROR);
    std::vector<double> wiki_not_div_4_example({1, 3, 5, 7, 9, 11, 13, 15, 17});
    EXPECT_NEAR(stats::interquartile_mean(wiki_not_div_4_example.begin(), wiki_not_div_4_example.end()), 9, ERROR);
}


int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/**

TEST_CASE("bootstrap") {
	std::vector<double> sample_data({0.1, 0.2, 0.3, 0.4, 0.5});

	SUBCASE("resample") {
		auto sample = moirai::resample(sample_data.begin(), sample_data.end());

		bool contains;
		for(auto itr = sample.begin(); itr != sample.end(); ++itr) {
			contains = false;
			for(auto itr2 = sample_data.begin(); itr2 != sample_data.end(); ++itr2) {
				if(*itr == *itr2) {
					contains = true;
					break;
				}
			}

			CHECK(contains == true);
		}
	}

	SUBCASE("build_historgram") {
		using SDType = std::vector<double>::value_type;
		using Itr = std::vector<double>::iterator;
		std::function<SDType(Itr, Itr)> mid = moirai::median<Itr>;
		auto histogram = moirai::build_histogram<Itr, decltype(mid)>(sample_data.begin(), sample_data.end(), mid, 1000);

		CHECK(histogram.size() == 1000);

		// octave.
		// for(auto p : histogram) {
		// 	std::cout << std::setw(2)
          //         << p.first << ' ' << std::string(p.second / 20, '*') << '\n';
		// }
	}

	SUBCASE("bootstrap") {
		using SDType = std::vector<double>::value_type;
		using Itr = std::vector<double>::iterator;
		std::function<SDType(Itr, Itr)> mid = moirai::median<Itr>;
		auto conf = moirai::bootstrap<Itr, decltype(mid)>(sample_data.begin(), sample_data.end(), mid, 0.95, 100000);

		CHECK(conf.confidence == 0.95);
		std::cout << "lower_bound: " << conf.lower_bound << ", upper_bound: " << conf.upper_bound << std::endl;
		// verify with octave maybe?
	}
}
 */