#ifndef STATS_STATS_HPP
#define STATS_STATS_HPP

/**
 * TODO:
 * 1.) Implement multithreading
 * 2.) Implement functions using Eigen::MatrixX(i,d,f) and Eigen::VectorX(i,d,f)
 * 3.) Write documentation for each funciton
 * 4.) Test skewness functions.
 * 5.) Iterative algorithms (updates with every new element or collection of elements)
 * 6.) Write Kendall rank correlation coefficient algorithm
 * 7.) Write Spearmans' rank correlation coefficient algorithm
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
     * Tested.
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
     * Tested.
     * computes the arithmetic mean of an Eigen::VectorXd
     * @param v: an Eigen::VectorXd object
     * @return the arithmetic mean of the data set represented by the Eigen::VectorXd
     */
    double mean(const Eigen::VectorXd& v) noexcept {
        return v.mean();
    }

    /// Tested.
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
     * Tested.
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

    /**
     * Tested.
     * computes the absolute error between two ints
     * @tparam T type of int
     * @param approx_val the approximated val
     * @param actual_val the actual or theoretical val
     * @return the absolute error (ie. difference) between the two vals
     */
    template <class T>
    constexpr auto abs_err(const T approx_val, const T actual_val) noexcept ->
    std::enable_if_t< (std::is_integral<T>::value || std::is_floating_point<T>::value) &&
                      !std::is_unsigned<T>::value, T> {
        return internal::abs(approx_val - actual_val);
    }

    /**
     * Tested.
     * computes the relative error between two values
     * @tparam T the type (integer or float)
     * @param approx_val the approximate value
     * @param actual_val the actual or theoretical val
     * @return the relative error between the two values
     */
    template <class T, class = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value>::type>
    constexpr auto rel_err(const T approx_val, const T actual_val) {
        assert((actual_val != 0));
        if(actual_val == 0) {
            throw std::logic_error("Attempting to divide by zero in rel_err function");
        }

        return (abs_err(approx_val, actual_val) / internal::abs((double)actual_val));
    }

    /**
     * Tested.
     * computes the percent error between two values
     * @tparam T the type (integer or float)
     * @param approx_val the approximate value
     * @param actual_val the actaul or theoretical value
     * @return the relative error between the two values
     */
    template <class T, class = typename std::enable_if<std::is_integral<T>::value || std::is_floating_point<T>::value>::type>
    constexpr auto perc_err(const T approx_val, const T actual_val) {
        return rel_err(approx_val, actual_val) * 100.0;
    }

    /**
     * Tested.
     * finds the max value in the dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the max value in the dataset
     */
    template <class Iterator>
    constexpr auto max(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attemting to find max of empty set.");
        }
        return *std::max_element(first, last);
    }

    /**
     * Tested.
     * finds the max value in the Eigen::VectorXd
     * @param vec the Eigen::VectorXd
     * @return the max value
     */
    double max(const Eigen::VectorXd& vec) {
        return vec.maxCoeff();
    }

    /**
     * Tested.
     * finds the min value in a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the min element in the dataset
     */
    template <class Iterator>
    constexpr auto min(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find min of empty set.");
        }
        return *std::min_element(first, last);
    }

    /**
     * Tested.
     * finds the min value in the Eigen::VectorXd
     * @param vec the Eigen::VectorXd
     * @return the min value
     */
    double min(const Eigen::VectorXd& vec) {
        return vec.minCoeff();
    }

    /**
     * Tested.
     * finds the median value in a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last poitner to the last element in the container
     * @return the median
     */
    template <class Iterator>
    constexpr auto median(Iterator first, Iterator last) {
        assert(first != last);
        if (first == last) {
            throw std::logic_error("Attempting to find median of empty set.");
        }

        auto size = std::distance(first, last);

        if (size % 2) {
            /// if there are an odd number of elements, return the middle element
            auto middle = first + size / 2;
            std::nth_element(first, middle, last);
            return *(middle);
        } else {
            /// if there are an even number of elements, return the average of the middle two
            auto middle_plus_one = first + size / 2;
            std::nth_element(first, middle_plus_one, last);
            /// Overflows if you add the two values then divide.
            return *(middle_plus_one) / 2 + *(middle_plus_one - 1) / 2;
        }
    }

    /// Rewrite function to not call stl
    /**
     * Tests failing.
     * finds the median value of a Eigen::VectorXd
     * @param vec the Eigen::VectorXd
     * @return the median
     */
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

    /// probably a better implementation of this
    /**
     * Tested.
     * finds the mode of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the mode
     */
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

    /**
     * Tested.
     * finds the mean of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the mean
     */
    template <class Iterator>
    constexpr auto mean(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find arithmetic mean of empty set.");
        }
        auto acc = std::accumulate(first, last, 0.0);
        return acc / (decltype(acc))std::distance(first, last);
    }

    /**
     * Tested.
     * finds the range of a dataset (max - min)
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the range
     */
    template <class Iterator>
    constexpr auto range(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find range of empty set.");
        }
        return stats::max(first, last) - stats::min(first, last);
    }

    /**
     * Tested.
     * computes the quartiles of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the quartiles
     */
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

    /**
     * Tested.
     * computes the interquartile mean of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the interquartile mean
     */
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

    /**
     * Tested.
     * computes the interquartile range of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the interquartile range
     */
    template <class Iterator>
    constexpr auto interquartile_range(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find interquartile range of empty set.");
        }
        auto [q1, q2, q3] = quartiles(first, last);
        return q3 - q1;
    }

    /**
     * Tested.
     * computes the outliers of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the outliers
     */
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

    /**
     * Tested.
     * computes the median absolute deviation of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the median absolute deviation
     */
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

    /**
     * Tested.
     * computes the variance of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the variance
     */
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

    /**
     * Not Tested.
     * computes the variance of a dataset given its mean
     * @tparam Iterator
     * @param first
     * @param last
     * @param mean
     * @return the variance
     */
    template <class Iterator>
    constexpr auto var(Iterator first, Iterator last, typename std::iterator_traits<Iterator>::value_type mean) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find variation of empty set.");
        }

        auto N = std::distance(first, last);
        double acc = 0.0;

        for (int i = 0; i < N; i++) {
            auto diff = *(first + i) - mean;
            acc += diff * diff;
        }

        return acc / (N - 1);
    }

    /**
     * Tested.
     * finds the variance of a Eigen::VectorXd
     * @param vec the Eigen::VectorXd
     * @return the variance
     */
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

    /**
     * Not Tested.
     * finds the variance of a Eigen::MatrixXd
     * @param vec the Eigen::MatrixXd
     * @return the variance
     */
    auto var(const Eigen::MatrixXd& m) {
        Eigen::VectorXd mu(m.rows());
        for (int i = 0; i < m.rows(); i++) {
            mu(i) = var(static_cast<Eigen::VectorXd>(m.row(i)));
        }
        return mu;
    }

    /**
     * Tested.
     * computes the standard deviation of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the standard deviation
     */
    template <class Iterator>
    constexpr auto std_dev(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find standard deviation of empty set.");
        }
        return std::sqrt(var(first, last));
    }

    template <class Iterator>
    constexpr auto std_dev(Iterator first, Iterator last, typename std::iterator_traits<Iterator>::value_type mean) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find standard deviation of empty set.");
        }
        return std::sqrt(stats::var(first, last, mean));
    }

    /**
     * Not Tested.
     * finds the standard deviation of a Eigen::VectorXd
     * @param vec the Eigen::VectorXd
     * @return the standard deviation
     */
    double std_dev(const Eigen::VectorXd& v) {
        return std::sqrt(var(v));
    }

    /**
     * Not Tested.
     * finds the standard deviation of a Eigen::MatrixXd
     * @param vec the Eigen::MatrixXd
     * @return the standard deviation
     */
    Eigen::VectorXd std_dev(const Eigen::MatrixXd& m) {
        return static_cast<Eigen::VectorXd>(var(m).array().sqrt());
    }

    /**
     * Tested.
     * computes the quartile deviation of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the quartile deviation
     */
    template <class Iterator>
    constexpr auto quartile_dev(Iterator first, Iterator last) {
        assert(first != last);
        if(first == last) {
            throw std::logic_error("Attempting to find quartile deviation of empty set");
        }
        return interquartile_range(first, last) / 2.0;
    }

    /**
     * Not Tested!
     * computes the mode skewness of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the mode skewness
     */
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

    /**
     * Tests failing.
     * computes the ex kurtosis of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the ex kurtosis
     */
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

        return g1 / (size * std::pow(s, 4)) - 3.0;
    }

    /**
     * Not Tested.
     * computes the Pearson correlation coefficient of a dataset
     * @tparam Iterator the type of elements in the dataset
     * @param first pointer to the first element in the container
     * @param last pointer to the last element in the container
     * @return the Pearson correlation coefficient
     */
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
        auto s1 = std_dev(first1, last1, mu1);
        auto s2 = std_dev(first2, last2, mu2);

        auto s = std::transform_reduce(first1, last1, first2, std::plus<ret_type>(), [=](auto i1, auto i2) {
            return (i1 - mu1) * (i2 - mu2);
        });

        return s / (s1 * s2);
    }

    /**
     * Tested.
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
     * Not Tested.
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
        auto var1 = stats::var(x_first, x_last, mean1);
        auto var2 = stats::var(y_first, y_last, mean2);

        /// Compute the sizes of the two populations
        auto size1 = std::distance(x_first, x_last);
        auto size2 = std::distance(y_first, y_last);

        /// Apply the t-statistic formula (https://en.wikipedia.org/wiki/T-statistic)
        auto mean_combined = (size1 / (size1 + size2)) * mean1  + (size2 / (size1 + size2)) * * size1;
        auto t = (mean1 - mean2) / gcem::sqrt(var1 / size1 + var2 / size2);
    }

    /**
     * Tested.
     * computes the harmonic mean of a dataset
     * @tparam Iterator an iterator of the collection storing the data sets
     * @param first pointer to the first element in the container
     * @param last the last iterator in the vector of vectors
     * @return harmonic mean
     */
    template <class Iterator>
    constexpr auto hmean(Iterator first, Iterator last) {
        using Type = typename std::iterator_traits<Iterator>::value_type;
        auto ret = std::accumulate(first, last, 0.0, [](Type acc, Type x) {
            return acc += 1 / x;
        });
        return ret / std::distance(first, last);
    }
};


#endif //ATROPOS_STATS_HPP