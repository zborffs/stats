//
// Created by Zach Bortoff on 2020-03-02.
//

/**
 * TODO
 * 1.) DONE - Implement analogous algorithms with/without using Eigen
 * 2.) Test bootstrapping results against theoretical, analytically-obtained results.
 * 3.) Benchmark differences between implementations (std:: vs. Eigen::). Also compute respective memory footprints and
 *     accuracy / precision.
 * 4.) Implement different bootstrapping methods (--Case Resampling--, Bayesian Bootstrapping, Smooth Bootstrap,
 *     Parametric Bootstrap, Resampling Residuals, Gaussian Process Regression Bootstrapping, and Wild Bootstrapping
 * 5.) Implement streaming (incremental) algorithms; i.e. suppose the data set is growing can we divide and conquuer
 *     to only solve part of the problem then add that to the previous result (or something)
 * 6.) Improve the confidence interval calculations by brushing up on theory.
 * 7.) Write documentation.
 */

#ifndef STATS_BOOTSTRAP_HPP
#define STATS_BOOTSTRAP_HPP

/// Standard Library Includes
#include <iterator>
#include <random>
#include <tuple>
#include <initializer_list>
#include <functional>
#include <vector>

/// External Library Includes
#include <Eigen/Dense>

/// Internal Library Includes
#include "stats.hpp"

namespace stats {

    /// One could easily abuse this class if they don't perform operations in the right order.
    template <class Itr, class RetType>
    class Bootstrap {
    protected:
        using ValueType = typename std::iterator_traits<Itr>::value_type;

        std::vector<ValueType> data_set_;
        std::vector<ValueType> resampled_data_;
        std::vector<std::function<RetType(Itr, Itr)> > f_;
        std::vector<ValueType> res_;
        std::vector<std::tuple<double, double> > confidence_interval_;
        std::vector<double> t_stat_;

    public:
        Bootstrap(Itr first, Itr last) : data_set_(first, last), resampled_data_(std::distance(first, last)) {}
        Bootstrap(Itr first, Itr last, std::vector<std::function<RetType(Itr, Itr)> >& f) : data_set_(first, last), resampled_data_(std::distance(first, last)), f_(f), res_(f_.size()) {
            for (int i = 0; i < f_.size(); i++) {
                res_.push_back(f_[i](first, last));
            }
        }
        Bootstrap(Itr first, Itr last, std::initializer_list<std::function<RetType(Itr, Itr)> >& l) : data_set_(first, last), resampled_data_(std::distance(first, last)), f_(l), res_(f_.size()) {
            for (int i = 0; i < f_.size(); i++) {
                res_.push_back(f_[i](first, last));
            }
        }

        /**
         * computes the confidence interval necessary to achieve a desired confidence
         * @param num_iteration the number of iterations for the monte carlo simulation
         * @param desired_confidence the minimum desired confidence interval for the estimator over the data set
         */
        void monte_carlo(const int num_iteration, const double desired_confidence = 0.95) {
            assert(desired_confidence > 0.0 && desired_confidence < 1.0);
            if (desired_confidence <= 0.0 || desired_confidence >= 1.0) {
                throw std::logic_error(std::string("The desired confidence is outside the allowable bounds: " + std::to_string(desired_confidence)));
            }

            std::vector<std::vector<ValueType> > est_resample(num_iteration);

            for (int i = 0; i < num_iteration; i++) {
                resample();
                est_resample[i].reserve(f_.size());
                for (int j = 0; j < f_.size(); j++) {
                    est_resample[i].push_back(f_[i](resampled_data_.begin(), resampled_data_.end()));
                }
            }

            compute_conf_interval(est_resample, desired_confidence);
        }

        std::vector<std::tuple<double, double> > confidence_interval() {
            return confidence_interval_;
        }

        [[nodiscard]] std::vector<std::tuple<double, double> > confidence_interval() const {
            return confidence_interval_;
        }

        std::vector<double> t_stat() {
            return t_stat_;
        }

        [[nodiscard]] std::vector<double> t_stat() const {
            return t_stat_;
        }

        void add_function(std::function<RetType(Itr, Itr)>& f) {
            f_.push_back(f);
        }

        void erase(int i) {
            f_.erase(i);
        }

    protected:

        /**
         * randomly resamples a original data set with replacement
         */
        void resample() {
            assert(data_set_.begin() != data_set_.end());
            if (data_set_.begin() == data_set_.end()) {
                throw std::logic_error("Attempting to resample empty set.");
            }

            resampled_data_.clear();

            auto size = std::distance(data_set_.begin(), data_set_.end());
            int index;
            std::random_device r_device;
            std::mt19937_64 prn_gen(r_device());
            std::uniform_int_distribution<> distro(0, size - 1);

            for (int i = 0 ; i < size; i++) {
                index = distro(prn_gen);
                assert(index >= 0 && index < size);

                resampled_data_.push_back(*(data_set_.begin() + index));
            }

            assert(resampled_data_.size() == size);
        }

        /**
         * computes the confidence intervals to satisfy a minimum desired confidence
         * @param resamp_est_first an iterator to the first element of a vector containing the result of each estimator
         * @param resamp_est_last an iterator to the last element of a vector containing the result of each estimators
         * @param confidence the desired confidence percentage (p-value)
         */
        void compute_conf_interval(std::vector<std::vector<ValueType> >& est_resample, const double confidence) {
            // This method implicitly assume normal distributions.
            double inv_cdf = stats::inv_normal_cdf(confidence);
            for (int i = 0; i < res_.size(); i++) {
                auto s = std_dev(est_resample[i].begin(), est_resample[i].end());
                auto lower_bound = res_[i] - inv_cdf * s;
                auto upper_bound = res_[i] + inv_cdf * s;
                confidence_interval_.push_back(std::make_tuple(lower_bound, upper_bound));
            }
        }
    };

    /// One could easily abuse this class if they don't perform operations in the right order.
    template <>
    class Bootstrap<Eigen::VectorXd, double> {
        using Vec = Eigen::VectorXd;

        Vec data_set_;
        Vec resampled_data_;
        std::vector<std::function<double(Vec)> > f_;
        Vec res_;
        Vec lower_conf_int_;
        Vec upper_conf_int_;
        Vec t_stat_;

    public:
        Bootstrap<Eigen::VectorXd>(Vec& data_set) : data_set_(data_set), resampled_data_(data_set.size()) {}
        Bootstrap<Eigen::VectorXd>(Vec& data_set, std::vector<std::function<double(Vec)> >& f) : data_set_(data_set), resampled_data_(data_set.size()), f_(f), res_(f_.size()), lower_conf_int_(f_.size()), upper_conf_int_(f_.size()), t_stat_(f_.size()) {
            resampled_data_.setZero();
            res_.setZero();
            lower_conf_int_.setZero();
            upper_conf_int_.setZero();
            t_stat_.setZero();

            for (int i = 0 ; f_.size(); i++) {
                res_(i) = f_[i](data_set_.array());
            }
        }
        Bootstrap<Eigen::VectorXd>(Vec& data_set, std::initializer_list<std::function<double(Vec)> > l) : data_set_(data_set), resampled_data_(data_set.size()), f_(l), res_(f_.size()), lower_conf_int_(f_.size()), upper_conf_int_(f_.size()), t_stat_(f_.size()) {
            resampled_data_.setZero();
            res_.setZero();
            lower_conf_int_.setZero();
            upper_conf_int_.setZero();
            t_stat_.setZero();

            for (int i = 0 ; f_.size(); i++) {
                res_(i) = f_[i](data_set_.array());
            }
        }

        /**
         * computes the confidence interval necessary to achieve a desired confidence
         * @param num_iteration the number of iterations for the monte carlo simulation
         * @param desired_confidence the minimum desired confidence interval for the estimator over the data set
         */
        void monte_carlo(const int num_iteration, const double desired_confidence = 0.95) {
            assert(desired_confidence > 0.0 && desired_confidence < 1.0);
            if (desired_confidence <= 0.0 || desired_confidence >= 1.0) {
                throw std::logic_error(std::string("The desired confidence is outside the allowable bounds: " + std::to_string(desired_confidence)));
            }

            Eigen::MatrixXd est_resample(f_.size(), num_iteration);
            assert(f_.size() == est_resample.rows() && data_set_.size() == est_resample.cols());

            for (int i = 0; i < num_iteration; i++) {
                resample();
                for (int j = 0; j < f_.size(); j++) {
                    est_resample(i) = f_[j](resampled_data_);
                }
            }

            compute_conf_interval(est_resample, desired_confidence);
        }

        std::tuple<Vec, Vec> confidence_interval() {
            return std::make_tuple(lower_conf_int_, upper_conf_int_);
        }

        [[nodiscard]] std::tuple<Vec, Vec> confidence_interval() const {
            return std::make_tuple(lower_conf_int_, upper_conf_int_);
        }

        Vec t_stat() {
            return t_stat_;
        }

        [[nodiscard]] Vec t_stat() const {
            return t_stat_;
        }

        void add_function(std::function<double(Vec)>& f) {
            f_.push_back(f);
        }

        void erase(int i) {
            f_.erase(i + f_.begin());
        }

    protected:

        /**
         * randomly resamples a original with replacement
         */
        void resample() {
            (resampled_data_.setRandom() * (data_set_.size() - 1)).array().floor();
        }

        /**
         * computes the confidence intervals to satisfy a minimum desired confidence
         * @param est_resample the estimated resample matrix
         * @param confidence the minimum confidence interval (p-value)
         */
        void compute_conf_interval(Eigen::MatrixXd& est_resample, const double confidence) {
            // This method implicitly assume normal distributions.
            double inv_cdf = stats::inv_normal_cdf(confidence);
            auto s = std_dev(est_resample);
            lower_conf_int_ = res_ - inv_cdf * s;
            upper_conf_int_ = res_ + inv_cdf * s;
        }
    };

    /**
     * An algorithm for comparing the means of two independent sets by bootstrapping
     * @tparam Iterator an iterator of the collection storing the data sets
     * @param first1 an iterator to the beginning of the first data set
     * @param last1 an iterator to the end of the first data set
     * @param first2 an iterator to the beginning of the second data set
     * @param last2 an iterator to the end of the second data set
     * @param n the number of iterations to perform to bootstrap
     * @return the p-value comparing the means of the two data sets
     */
    template <class Iterator>
    double test_bootstrap_hypothesis(Iterator first1, Iterator last1, Iterator first2, Iterator last2, int n = 1) {
        assert(first1 != last1 && first2 != last2);
        if (first1 == last1 || first2 == last2) {
            throw std::logic_error("Attempting to test the hypothesis of empty set.");
        }

        /// Compute the t-statistic
        /// Compute the means of the two populations
        auto mean1 = stats::mean(first1, last1);
        auto mean2 = stats::mean(first2, last2);

        /// Compute the variations of the two popuations
        auto var1 = stats::var(first1, last1);
        auto var2 = stats::var(first2, last2);

        /// Compute the sizes of the two populations
        auto size1 = std::distance(first1, last1);
        auto size2 = std::distance(first2, last2);

        /// Apply the t-statistic formula (https://en.wikipedia.org/wiki/T-statistic)
        auto mean_combined = (size1 / (size1 + size2)) * mean1  + (size2 / (size1 + size2)) * * size1;
        auto t = (mean1 - mean2) / gcem::sqrt(var1 / size1 + var2 / size2);

        using itr_type = typename std::iterator_traits<Iterator>::value_type;
        std::vector<itr_type> xnew(size1);
        std::vector<itr_type> ynew(size2);

        std::transform(first1, last1, xnew.begin(), [=](itr_type x) {
            return x - mean1 + mean_combined;
        });
        std::transform(first2, last2, ynew.begin(), [=](itr_type y) {
            return y - mean2 + mean_combined;
        });

        if (n <= 0) {
            n = 1;
        }

        std::vector<double> t_star_vec(n);

        for (int i = 0; i < n; i++) {
            auto x_star = resample(xnew.begin(), xnew.end());
            auto y_star = resample(ynew.begin(), ynew.end());

            auto x_star_mean = stats::mean(x_star.begin(), x_star.end());
            auto y_star_mean = stats::mean(y_star.begin(), y_star.end());
            auto x_star_s = stats::var(x_star.begin(), x_star.end());
            auto y_star_s = stats::var(y_star.begin(), y_star.end());

            auto t_star = (x_star_mean - y_star_mean) / (gcem::sqrt(x_star_s / size1 + y_star_s / size2));
            t_star_vec.push_back(t);
        }


        return std::accumulate(t_star_vec.begin(), t_star_vec.end(), [=](auto t_star) {
            return t_star >= t;
        }) / n;
    }

};

#endif //ATROPOS_BOOTSTRAP_HPP
