//
// Created by Zach Bortoff on 2020-03-21.
//

#include <benchmark/benchmark.h>
#include <gsl/gsl_statistics_double.h>
#include <Eigen/Dense>
#include "stats.hpp"
#include <random>

static void BM_GSL_Mean(benchmark::State& state) {
    unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine e(seed);
    std::normal_distribution<double> d(186.31239557544, 241.61891865);
    double data[state.range(0)];

    for (int i = 0; i < state.range(0); i++) {
        data[i] = d(e);
    }

    for (auto _ : state) {
        gsl_stats_mean(data, 1, state.range(0));
    }
}

// Register the function as a benchmark
BENCHMARK(BM_GSL_Mean)->Range(1024, 262144);
//BENCHMARK(BM_GSL_Mean);

// Define another benchmark
static void BM_STL_Mean(benchmark::State& state) {
    unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine e(seed);
    std::normal_distribution<double> d(186.31239557544, 241.61891865);
    std::vector<double> data(state.range(0));

    for (int i = 0; i < state.range(0); i++) {
        data[i] = d(e);
    }

    for (auto _ : state) {
        stats::mean(data.begin(), data.end());
    }
}
BENCHMARK(BM_STL_Mean)->Range(1024, 262144);

// Define another benchmark
static void BM_Eigen_Mean(benchmark::State& state) {
    unsigned int seed = std::chrono::steady_clock::now().time_since_epoch().count();
    std::default_random_engine e(seed);
    std::normal_distribution<double> d(186.31239557544, 241.61891865);
    Eigen::RowVectorXd row_data(state.range(0));
    for (int i = 0; i < state.range(0); i++) {
        row_data(i) = d(e);
    }

    Eigen::VectorXd data = row_data.transpose();
    for (auto _ : state) {
        stats::mean(data);
    }
}
BENCHMARK(BM_Eigen_Mean)->Range(1024, 262144);
//BENCHMARK(BM_Eigen_Mean);

BENCHMARK_MAIN();