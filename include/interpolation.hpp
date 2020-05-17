//
// Created by Zach Bortoff on 2020-05-15.
//

#ifndef STATS_INTERPOLATION_HPP
#define STATS_INTERPOLATION_HPP

#include "stats.hpp"
#include <Eigen/Dense>

namespace st {
    std::vector<double> lin_interp(const std::vector<double>& x, const std::vector<double>& y)  {
        assert(x.size() == y.size());
        std::vector<double> theta{};
        return theta;
    }

    Eigen::VectorXd lin_interp(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        assert(x.size() == y.size());
        Eigen::VectorXd theta;
        return theta;
    }

    double lin_regress(const std::vector<double>& theta, const std::vector<double>& x, const std::vector<double>& y) {
        assert(x.size() == y.size());
        return 0.0;
    }

    double lin_regress(const Eigen::VectorXd& theta, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        assert(x.size() == y.size());
        return 0.0;
    }

    std::vector<double> quad_interp(const std::vector<double>& x, const std::vector<double>& y) {
        assert(x.size() == y.size());
        std::vector<double> theta{};
        return theta;
    }

    Eigen::VectorXd quad_interp(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        assert(x.size() == y.size());
        Eigen::VectorXd theta;
        return theta;
    }

    double quad_regress(const std::vector<double>& theta, const std::vector<double>& x, const std::vector<double>& y) {
        assert(x.size() == y.size());
        return 0.0;
    }

    double quad_regress(const Eigen::VectorXd& theta, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        assert(x.size() == y.size());
        return 0.0;
    }

    std::vector<double> logistic_interp(const std::vector<double>& x, const std::vector<double>& y) {
        assert(x.size() == y.size());
        std::vector<double> theta{};
        return theta;
    }

    Eigen::VectorXd logistic_interp(const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        assert(x.size() == y.size());
        Eigen::VectorXd theta;
        return theta;
    }

    double logistic_regress(const std::vector<double>& theta, const std::vector<double>& x, const std::vector<double>& y) {
        assert(x.size() == y.size());
        return 0.0;
    }

    double logistic_regress(const Eigen::VectorXd& theta, const Eigen::VectorXd& x, const Eigen::VectorXd& y) {
        assert(x.size() == y.size());
        return 0.0;
    }
};

#endif //STATS_INTERPOLATION_HPP
