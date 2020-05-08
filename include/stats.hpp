#ifndef ATROPOS_STATS_HPP
#define ATROPOS_STATS_HPP

/**
 * TODO:
 * 1.) Implement multithreading
 * 2.) Implement functions using Eigen::MatrixX(i,d,f) and Eigen::VectorX(i,d,f)
 * 3.) Write documentation for each funciton
 * 4.) Test skewness functions.
 */

/// Standard Library Includes
#include <algorithm>
#include <functional>
#include <numeric>
#include <unordered_map>
#include <unordered_set>
#include <utility>

/// External Library Includes
#include <Eigen/Eigen>
#include <parts/gcem.hpp>

namespace stats {
    /**
     * computes the inverse of the normal cdf function evaluated at a given percent
     * @param p the percent confidence
     * @return the inverse of the normal cdf function evaluated for a given percent
     */
    constexpr double inv_normal_cdf(const double p) {
        assert(p >= 0 && p <= 1);
        if(p < 0 && p > 1) {
            throw std::logic_error("p value of inverse normal cdf must be [0, 1].");
        }
        // modified equation from wikipedia. this renders the right answer.
        return gcem::sqrt(2) * gcem::erf_inv(p);
    }

    /**
     * computes the arithmetic mean of an Eigen::VectorXd
     * @param v: an Eigen::VectorXd object
     * @return the arithmetic mean of the data set represented by the Eigen::VectorXd
     */
    double mean(const Eigen::VectorXd& v) noexcept {
        return v.mean();
    }

    namespace internal {

        /*
         * These functions are courtesy of user: jan.sende from
         * stackoverflow.com
         * https://stackoverflow.com/questions/30084577/ambiguous-call-to-abs/30085929
         * These functions help circumvent an error produced by the
         * abs_err() function below, in which an "ambiguous call" to
         * "std::abs" was thrown during compilation.
         */

        /**
         * returns the absolute value
         * @param  value any integral or float
         * @return       absolute value of input number
         */
        template<class T>
        constexpr auto abs(T value) -> std::enable_if_t<std::is_unsigned<T>::value, T> {
            return value;
        }

        template<class T>
        constexpr auto abs(T value) -> std::enable_if_t<std::is_floating_point<T>::value, T> {
            return std::fabs(value);
        }

        template<class T>
        constexpr auto abs(T value) -> std::enable_if_t<std::is_same<T, int>::value, T> {
            return std::abs(value);
        }

        template<class T>
        constexpr auto abs(T value) -> std::enable_if_t<std::is_same<T, long>::value, T> {
            return std::labs(value);
        }

        template<class T>
        constexpr auto abs(T value) -> std::enable_if_t<std::is_same<T, long long>::value, T> {
            return std::llabs(value);
        }

        template<class T>
        constexpr auto abs(T value) -> std::enable_if_t<std::is_signed<T>::value &&
                                                        !std::is_floating_point<T>::value &&
                                                        !std::is_same<T, int>::value &&
                                                        !std::is_same<T, long>::value &&
                                                        !std::is_same<T, long long>::value, T> {
            return std::abs(value);
        }
    };

    /**
     * computes the absolute error between two values
     * @tparam T a real number or an integer
     * @param approx_val the approximate / experimental value
     * @param actual_val the actual / theoretical value
     * @return the absolute error of the two numbers
     */
    template <class T>
    constexpr auto abs_err(const T approx_val, const T actual_val) noexcept ->
    std::enable_if_t<(std::is_integral<T>::value || std::is_floating_point<T>::value) &&
                     std::is_unsigned<T>::value, T> {
        // unsigned numbers can never be negative, obviously, so there
        // is no need to check for that. Instead, we just make sure that
        // the difference between the numbers is a positive number of
        // checking the size of each value. That way, when the function
        // inevitably returns another unsigned number, you don't get a
        // wrap around error.
        if(approx_val > actual_val) {
            return approx_val - actual_val;
        } else {
            return actual_val - approx_val;
        }
    }

    template <class T>
    constexpr auto abs_err(const T approx_val, const T actual_val) noexcept ->
    std::enable_if_t< (std::is_integral<T>::value || std::is_floating_point<T>::value) &&
                      !std::is_unsigned<T>::value, T> {
        return internal::abs(approx_val - actual_val);
    }

    template <class T, class = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value>::type>
    constexpr auto rel_err(const T approx_val, const T actual_val) {
        assert((actual_val != 0));
        if(actual_val == 0) {
            throw std::logic_error("Attempting to divide by zero in rel_err function");
        }

        return (abs_err(approx_val, actual_val) / internal::abs((double)actual_val));
    }

    template <class T, class = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value>::type>
    constexpr auto perc_err(const T approx_val, const T actual_val) {
        return rel_err(approx_val, actual_val) * 100.0;
    }

    template <class Iterator>
    constexpr auto max(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attemting to find max of empty set.");
        }
        return *std::max_element(first, last);
    }

    double max(const Eigen::VectorXd& vec) {
        return vec.maxCoeff();
    }

    template <class Iterator>
    constexpr auto min(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find min of empty set.");
        }
        return *std::min_element(first, last);
    }

    double min(const Eigen::VectorXd& vec) {
        return vec.minCoeff();
    }

    template <class Iterator>
    constexpr auto median(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find median of empty set.:");
        }

        auto size = std::distance(first, last);

        if(size % 2) {
            auto middle = first + size / 2;
            std::nth_element(first, middle, last);
            return *(middle);
        } else {
            auto middle_plus_one = first + size / 2;
            std::nth_element(first, first + size / 2 + 1, last);
            return (*(middle_plus_one) + *(middle_plus_one - 1)) / 2;
        }
    }

    double median(Eigen::VectorXd& vec) {
        return stats::median(vec.data(), vec.data() + vec.size());
//        auto first = vec.data();
//        auto last = vec.data() + vec.size();
//        auto size = std::distance(first, last);
//
//        if(size % 2) {
//            auto middle = first + size / 2;
//            std::nth_element(first, middle, last);
//            return *(middle);
//        } else {
//            auto middle_plus_one = first + size / 2;
//            std::nth_element(first, first + size / 2 + 1, last);
//            return (*(middle_plus_one) + *(middle_plus_one - 1)) / 2;
//        }
    }

    template <class Iterator>
    constexpr auto mode(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find mode of empty set.");
        }

        using itr_type = typename std::iterator_traits<Iterator>::value_type;

        // might make more sense to just use a binary tree, or a heap,
        // then to sort it afterwards.
        auto size = std::distance(first, last);
        std::unordered_map<itr_type, int> hash_table;
        std::unordered_set<itr_type> maxima;
        hash_table.reserve(size);
        maxima.reserve(size);

        for(auto v = first; v != last; ++v) {
            hash_table[*v]++;
        }

        int max = hash_table.begin()->second;
        for (auto v = hash_table.begin(); v != hash_table.end(); ++v) {
            if (maxima.empty() || v->second == max) {
                maxima.insert(v->first);
            } else if(v->second > max) {
                max = v->second;
                maxima.clear();
                maxima.insert(v->first);
            }
        }

        return maxima;
    }

    template <class Iterator>
    constexpr auto mean(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find arithmetic mean of empty set.");
        }
        auto acc = std::accumulate(first, last, 0.0);
        return acc / (decltype(acc))std::distance(first, last);
    }

    template <class Iterator>
    constexpr auto range(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find range of empty set.");
        }
        return stats::max(first, last) - stats::min(first, last);
    }

    template <class Iterator>
    constexpr auto quartiles(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find quartiles of empty set.");
        }
        auto med = median(first, last);
        auto middle = std::partition(first, last, [med](decltype(med) v) {
            return v <= med;
        });

        typename std::iterator_traits<Iterator>::value_type q1;
        typename std::iterator_traits<Iterator>::value_type q3;
        if(std::distance(first, last) % 2) {
            q1 = median(first, middle - 1);
            q3 = median(middle, last);
        } else {
            q1 = median(first, middle);
            q3 = median(middle, last);
        }

        return std::make_tuple(q1, med, q3);
    }

    template <class Iterator>
    constexpr auto interquartile_mean(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find interquartile mean of empty set.");
        }
        using itr_type = typename std::iterator_traits<Iterator>::value_type;

        auto size = std::distance(first, last);
        std::sort(first, last);
        auto quartile_size = size / 4.0;
        auto perc = (4 - (size % 4)) / 4.0;

        int count = 0;
        auto f1 = first + (int)quartile_size;
        auto l1 = last - (int)quartile_size;
        int s = std::distance(f1, l1);
        auto acc = std::accumulate(f1, l1, 0.0, [&](itr_type acc, itr_type b) {
            if (count == 0 || count == s - 1) {
                count++;
                return acc + b * perc;
            }
            count++;
            return acc + b;
        });

        return acc / (2 * quartile_size);
    }

    template <class Iterator>
    constexpr auto interquartile_range(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find interquartile range of empty set.");
        }
        auto [q1, q2, q3] = quartiles(first, last);
        return q3 - q1;
    }

    template <class Iterator>
    constexpr auto outliers(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find outliers of empty set.");
        }
        using itr_type = typename std::iterator_traits<Iterator>::value_type;

        auto tuple = quartiles(first, last);
        auto q1 = std::get<0>(tuple);
        auto q3 = std::get<2>(tuple);
        auto iqr = q3 - q1;
        std::unordered_set<itr_type > ret;
        std::copy_if(first, last, std::inserter(ret, ret.begin()), [=](itr_type v) {
            return (v > q3 + iqr * 1.5) || (v < q1 - iqr * 1.5);
        });

        return ret;
    }

    template <class Iterator>
    constexpr auto median_abs_dev(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find median absolute deviation of empty set.");
        }
        using itr_type = typename std::iterator_traits<Iterator>::value_type;
        auto med = median(first, last);
        auto size = std::distance(first, last);

        std::vector<itr_type> new_vec;
        new_vec.reserve(size);
        std::transform(first, last, std::back_inserter(new_vec), [=](itr_type v) {
            return std::abs((itr_type)med - v);
        });

        return median(new_vec.begin(), new_vec.end());
    }

    template <class Iterator>
    constexpr auto var(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find variation of empty set.");
        }

        auto mu = mean(first, last);
        auto N = std::distance(first, last);
        double acc = 0.0;

        for (int i = 0; i < N; i++) {
            auto diff = *(first + i) - mu;
            acc += diff * diff;
        }

        return acc / (N - 1);
    }

    double var(const Eigen::VectorXd& v) {
        double mean = stats::mean(v);
        int N = v.size();
        double acc = 0.0;

        for (int i = 0; i < N; i++) {
            auto diff = v(i) - mean;
            acc += diff * diff;
        }

        return acc / (N - 1);
    }

    auto var(const Eigen::MatrixXd& m) {
        Eigen::VectorXd mu(m.rows());
        for (int i = 0; i < m.rows(); i++) {
            mu(i) = var(static_cast<Eigen::VectorXd>(m.row(i)));
        }
        return mu;
    }

    template <class Iterator>
    constexpr auto std_dev(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find standard deviation of empty set.");
        }
        return std::sqrt(var(first, last));
    }

    double std_dev(const Eigen::VectorXd& v) {
        return std::sqrt(var(v));
    }

    Eigen::VectorXd std_dev(const Eigen::MatrixXd& m) {
        return static_cast<Eigen::VectorXd>(var(m).array().sqrt());
    }

    template <class Iterator>
    constexpr auto quartile_dev(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find quartile deviation of empty set");
        }
        return interquartile_range(first, last) / 2.0;
    }

    template <class Iterator>
    constexpr auto mode_skewness(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find mode skewness of empty set.");
        }
        auto mo = mode(first, last);
        if(mo.size() != 1) {
            throw std::logic_error("The population sample isn't unimodal.");
        }
        auto me = mean(first, last);
        auto s = std_dev(first, last);

        return (me - mo[0].first) / s;
    }

    template <class Iterator>
    constexpr auto median_skewness(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find median skewness of empty set.");
        }
        auto me = mean(first, last);
        auto med = median(first, last);
        auto s = std_dev(first, last);

        return 3.0 * (me - med) / s;
    }

    template <class Iterator>
    constexpr auto quartile_skewness(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find quartile skewness of empty set.");
        }
        auto [q1, q2, q3] = quartiles(first, last);
        return (q3 + q1 - 2.0 * q2) / (q3 - q1);
    }

    template <class Iterator>
    constexpr auto ex_kurtosis(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find ex kurtosis of empty set.");
        }
        auto mu = mean(first, last);
        auto s = std_dev(first, last);

        std::vector<decltype(mu)> new_vec;
        new_vec.reserve(std::distance(first, last));

        std::transform(first, last, std::back_inserter(new_vec), [=](decltype(mu) x) {
            return std::pow((x - mu) / s, 2.0);
        });

        auto v = var(new_vec.begin(), new_vec.end());

        return v + 1.0 - 3.0;
    }

    template <class Iterator>
    constexpr auto pears_corr_coeff(Iterator first1, Iterator last1, Iterator first2, Iterator last2) {
        using ret_type = typename std::iterator_traits<Iterator>::value_type;
        if(first1 == last1 || first2 == last2) {
            throw std::logic_error("Attempting to find pearson correlation coefficient of empty set.");
        }
        auto size1 = std::distance(first1, last1);
        auto size2 = std::distance(first2, last2);

        assert(size1 > 0 && size2 > 0 && size1 == size2);
        if(size1 <= 0 || size2 <= 0 || size1 != size2) {
            throw std::logic_error("Comparing disjoint populations in Pearson Correlation Coefficient Function.");
        }

        // These lines are inefficient, because we end up computing the
        // arithemetic mean twice per population.
        auto mu1 = mean(first1, last1);
        auto mu2 = mean(first2, last2);
        auto s1 = std_dev(first1, last1);
        auto s2 = std_dev(first2, last2);

        auto s = std::transform_reduce(first1, last1, first2, std::plus<ret_type>(), [=](auto i1, auto i2) {
            return (i1 - mu1) * (i2 - mu2);
        });

        return s / (s1 * s2);
    }

    /**
     * computes the pooled relative repeatability of a vector of vectors
     * @tparam Iterator an iterator of the vector of vectors
     * @param first the first iterator in the vector of vectors
     * @param last the last iterator in the vector of vectors
     * @return the poooled relative repeatability of a vector of vectors
     *
     * References:
     * https://en.wikipedia.org/wiki/Repeatability
     * https://www.youtube.com/watch?v=ckcU4iPvhlg
     */
    template <class Iterator>
    constexpr auto repeatability(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find repeatability of empty set.");
        }

        double rsd_sum = 0.0;
        int deg_sum = 0;

        for (auto v = first; v != last; ++v) {
            auto f_i = v->begin();
            auto l_i = v->end();

            assert(first != last);
            if(f_i == l_i) {
                throw std::logic_error("Attempting to find repeatability of empty set.");
            }

            int deg_of_freedom = std::distance(f_i, l_i) - 1;
            rsd_sum += std::pow(stats::std_dev(f_i, l_i) / stats::mean(f_i, l_i), 2) * (deg_of_freedom);
            deg_sum += deg_of_freedom;
        }

        return sqrt(rsd_sum / deg_sum);
    }

    /**
     * computes the t-statistic between two data sets
     * @tparam Iterator an iterator of the collection storing the data sets
     * @param x_first an iterator to the beginning of the first data set
     * @param x_last an iterator to the end of the first data set
     * @param y_first an iterator to the beginning of the second data set
     * @param y_last an iterator to the end of the second data set
     * @return the t-statistic
     */
    template <class Iterator>
    constexpr auto t_statistic(Iterator x_first, Iterator x_last, Iterator y_first, Iterator y_last) {
        /// Compute the means of the two populations
        auto mean1 = stats::mean(x_first, x_last);
        auto mean2 = stats::mean(y_first, y_last);

        /// Compute the variations of the two popuations
        auto var1 = stats::var(x_first, x_last);
        auto var2 = stats::var(y_first, y_last);

        /// Compute the sizes of the two populations
        auto size1 = std::distance(x_first, x_last);
        auto size2 = std::distance(y_first, y_last);

        /// Apply the t-statistic formula (https://en.wikipedia.org/wiki/T-statistic)
        auto mean_combined = (size1 / (size1 + size2)) * mean1  + (size2 / (size1 + size2)) * * size1;
        auto t = (mean1 - mean2) / gcem::sqrt(var1 / size1 + var2 / size2);
    }

};


#endif //ATROPOS_STATS_HPP