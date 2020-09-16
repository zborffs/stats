//
// Created by Zach Bortoff on 2020-05-09.
//

#ifndef STATS_RANKING_HPP
#define STATS_RANKING_HPP

/**
 * TODO:
 * 1.) Implement multithreading
 * 2.) Compute things iteratively rather than all at once
 * 2a.) Currently Glicko2 is computed in batches. Is it possible to compute the intermediate steps while the games are
 *      being played, so that we don't have to perform all the computations at once? If so, how could we compute things
 *      in a multithreaded manner to improve the performance even further?
 * 3.) Run benchmark to test the two alternative implementations of hypothetical elo difference function.
 * 4.) Optimize iteration step in glicko2 algorithm.
 * 5.) Write documentation.
 */

/// Standard Library Includes
#include <utility>
#include <vector>
#include <iostream>
#include <cmath>
#include <unordered_map>

/// External Library Includes
#include "parts/gcem.hpp"
#include <Eigen/Dense>
#include <unsupported/Eigen/SpecialFunctions>

/// Internal Library Includes
#include "stats.hpp"

namespace st {
    struct Player {
        std::string name;
        int elo_rating;
        int elo_eff_num_games;
        int glicko_rating;
        int glicko_deviation;
        double glicko_volatility;

        explicit Player(std::string n = "No Name", const int er = 1200, const int eeng = 0, const int gr = 1500, const int gd = 350, const double gv = 0.06) : name(std::move(n)), elo_rating(er), elo_eff_num_games(eeng), glicko_rating(gr), glicko_deviation(gd), glicko_volatility(gv) {}

        friend std::ostream& operator<<(std::ostream& os, const Player& p) {
            os << "Player:" << std::endl;
            os << "  name: " << p.name << std::endl;
            os << "  elo: " << p.elo_rating << std::endl;
            os << "  glicko2: " << p.glicko_rating - 2 * p.glicko_deviation << "-" << p.glicko_rating + 2 * p.glicko_deviation << " (0.95, " << std::to_string(p.glicko_volatility) << ")" << std::endl;
            return os;
        }
    };

    struct Match {
        Player white;
        Player black;
        int start_sec; // starting time on the clock in seconds per player
        int delay_sec; // delay in seconds per player
        int incr_sec; // increment in seconds per player
        int duration_sec; // how long the match actually lasted in seconds in total (for both players)
        int result;
        std::string text;

        explicit Match(Player w, Player b, int start, int delay, int incr, int dur = 0, int res = 0, std::string t = "") : white(std::move(w)), black(b), start_sec(start), delay_sec(delay), incr_sec(incr), duration_sec(dur), result(res), text(t) {}

        friend std::ostream& operator<<(std::ostream& os, const Match& m) {
            os << "[White \"" << m.white.name << "\"]" << std::endl;
            os << "[Black \"" << m.black.name << "\"]" << std::endl;
            os << "[White ELO \"" << m.white.elo_rating << "\"]" << std::endl;
            os << "[Black ELO \"" << m.black.elo_rating << "\"]" << std::endl;
//                os << "[Match Time \"" << m.start_sec << ":" << m.delay_sec << ":+" << m.incr_sec << "\"]" << std::endl;
//                os << "[Duration \"" << m.duration_sec << "\"]" << std::endl;
            os << "[Result \"" << m.result << "\"]" << std::endl;
            os << std::endl;
            os << m.text << std::endl;
            return os;
        }
    };

    class Tournament {
    protected:
        using Matrix = Eigen::MatrixXd;
        using Vector = Eigen::VectorXd;

        std::vector<std::string> player_names_;
        Matrix elo_diff_;
        Vector elo_;
        Vector glicko_rating_;
        Vector glicko_deviation_;
        Vector glicko_volatility_;
        int num_matches_per_player_sq_;
        double glicko2_system_rating_; // 0.3-1.2 possibly even 0.2
        int glicko2_game_period_; // 15-20, but depends
        Vector num_games_;
        Matrix win_loss_matrix_;
        Matrix draw_matrix_;
        std::vector<Match> matches_{};


    public:
        Tournament(std::vector<Player> players, const int n, double gsr = 0.6, int glicko2_game_period = 17) : elo_diff_(players.size(), players.size()), elo_(players.size()), glicko_rating_(players.size()), glicko_deviation_(players.size()), glicko_volatility_(players.size()), num_matches_per_player_sq_(n), glicko2_system_rating_(gsr), glicko2_game_period_(glicko2_game_period), num_games_(0), win_loss_matrix_(players.size(), players.size()), draw_matrix_(players.size(), players.size()) {
            player_names_ = std::vector<std::string>(players.size());
            elo_diff_.setZero();
            win_loss_matrix_.setZero();
            draw_matrix_.setZero();
            num_games_.setZero();

            for (int i = 0; i < players.size(); i++) {
                player_names_.push_back(players[i].name);
                glicko_rating_(i) = players[i].glicko_rating;
                glicko_deviation_(i) = players[i].glicko_deviation;
                glicko_volatility_(i) = players[i].glicko_volatility;
                elo_(i) = players[i].elo_rating;

                for (int j = 0; j < players.size(); j++) {
                    if (i == j) {
                        continue;
                    }
                    elo_diff_(i, j) = players[i].elo_rating - players[j].elo_rating;
                }
            }
        }

        void start() {
            /// Everyone plays each other
            return;
        }

        bool publish_results(const std::string& file) {
            /// Dump all Match data into a file.
            return false;
        }

        void add_match(const st::Match& match) {
            int i = 0;
            int j = 0;

            /// inefficient: use a hashtable or bintree
            for (; i < player_names_.size(); i++) {
                if (player_names_[i] == match.white.name) {
                    break;
                }
            }

            assert(i != player_names_.size());
            if (i == player_names_.size()) {
                throw std::logic_error("The White player is not in this tournament!");
            }

            /// inefficient: use a hashtable or bintree
            for(; j < player_names_.size(); j++) {
                if (player_names_[j] == match.black.name) {
                    break;
                }
            }

            assert(j != player_names_.size());
            if (j == player_names_.size()) {
                throw std::logic_error("The Black player is not in this tournament!");
            }

            switch(match.result) {
                case -1:
                    win_loss_matrix_(j, i) += 1;
                    break;
                case 0:
                    draw_matrix_(i, j) += 1;
                    break;
                case 1:
                    win_loss_matrix_(i, j) += 1;
                    break;
            }

            num_games_(i) += 1;
            num_games_(j) += 1;
            matches_.push_back(match);
        }

        Matrix score_diff_matrix() {
            return (win_loss_matrix_ - win_loss_matrix_.transpose());
        }

        [[nodiscard]] Matrix score_diff_matrix() const {
            return (win_loss_matrix_ - win_loss_matrix_.transpose());
        }

        Matrix score_matrix() {
            return win_loss_matrix_ + (0.5 * draw_matrix_.transpose());
        }

        [[nodiscard]] Matrix score_matrix() const {
            return win_loss_matrix_ + (0.5 * draw_matrix_.transpose());
        }

        constexpr int num_games() {
            auto ret = num_games_.sum();
            assert(ret >= 0);
            return ret;
        }

        [[nodiscard]] constexpr int num_games() const {
            auto ret = num_games_.sum();
            assert(ret >= 0);
            return ret;
        }

        constexpr int num_games(int i) {
            assert(i < num_games_.size() && i >= 0);
            if (i < 0 || i >= num_games_.size()) {
                throw std::logic_error("Trying to find the number of games played by an invalid index.");
            }
            auto ret = num_games_(i);
            assert(ret >= 0);
            return ret;
        }

        [[nodiscard]] constexpr int num_games(int i) const {
            assert(i < num_games_.size() && i >= 0);
            if (i < 0 || i >= num_games_.size()) {
                throw std::logic_error("Trying to find the number of games played by an invalid index.");
            }
            auto ret = num_games_(i);
            assert(ret >= 0);
            return ret;
        }

        Matrix win_ratio_matrix() {
            return (static_cast<Matrix>(win_loss_matrix_) + 0.5 * static_cast<Matrix>(draw_matrix_)).array() / num_games_.array();
        }

        [[nodiscard]] Matrix win_ratio_matrix() const {
            return (static_cast<Matrix>(win_loss_matrix_) + 0.5 * static_cast<Matrix>(draw_matrix_)).array() / num_games_.array();
        }

        Matrix expected_win_ratio_matrix() {
            return Eigen::inverse((1.0 + Eigen::pow(10, (-1.0 / 400.0 * static_cast<Matrix>(elo_diff_).array()))));
        }

        [[nodiscard]] Matrix expected_win_ratio_matrix() const {
            return Eigen::inverse((1.0 + Eigen::pow(10, (-1.0 / 400.0 * static_cast<Matrix>(elo_diff_).array()))));
        }

        Matrix los_matrix() {
            Matrix diff_matrix = static_cast<Matrix>(score_diff_matrix());
            Matrix sum_times_2_matrix = 2.0 * (Matrix)(win_loss_matrix_ + win_loss_matrix_.transpose());
            Matrix inter = diff_matrix * static_cast<Matrix>(Eigen::rsqrt(sum_times_2_matrix.array())).transpose();
            auto erf = 1.0 + Eigen::erf(inter.array());
            return static_cast<Matrix>(0.5 * erf.array());
        }

        [[nodiscard]] Matrix los_matrix() const {
            Matrix diff_matrix = static_cast<Matrix>(score_diff_matrix());
            Matrix sum_times_2_matrix = 2.0 * static_cast<Matrix>(win_loss_matrix_ + win_loss_matrix_.transpose());
            Matrix inter = diff_matrix * static_cast<Matrix>(Eigen::rsqrt(sum_times_2_matrix.array())).transpose();
            auto erf = 1.0 + Eigen::erf(inter.array());
            return static_cast<Matrix>(0.5 * erf.array());
        }

        void update_glicko_ratings() {
            const double pi = gcem::acos(-1);
            const double convergeance_tolerance = 0.000001;
            double v = 0.0;
            const int n = player_names_.size();

            Vector mu(n); mu.setZero();
            Vector phi(n); phi.setZero();
            Vector sigma(n); sigma.setZero();
            Vector Delta(n); Delta.setZero();
            Matrix g(n, n); g.setZero();
            Matrix E(n, n); E.setZero();

            Vector glicko2_rating(n); glicko2_rating.setZero();
            Vector glicko2_deviation(n); glicko2_deviation.setZero();
            Vector glicko2_volatility(n); glicko2_volatility.setZero();

            /// Convert from Glicko System to Glicko2 System
            glicko2_rating = static_cast<Vector>((static_cast<Vector>(glicko_rating_).array() - 1500) / 173.7178);
            glicko2_deviation = static_cast<Vector>(glicko_deviation_) / 173.7178;

            g = static_cast<Matrix>(Eigen::rsqrt(1.0 + (3.0 * (glicko2_deviation / pi).array().square())));
            E = static_cast<Matrix>(1.0 / (1.0 + (-g * (glicko2_rating - glicko2_rating.transpose()).transpose()).array().exp()));

            v = (g * g.transpose() * E * static_cast<Matrix>(1.0 - E.array())).sum();
            v = 1 / v;

            Vector delta(n); delta.setZero();
            auto s = score_matrix();
            for (int i = 0; i < player_names_.size(); i++) {
                delta(i) = (g * (s - E).transpose()).sum();
            }

            delta = delta * v;
            Vector a(n); a.setZero();
            a = 2 * Delta.array().log();
            auto Delta_squared = Delta.array().square();
            auto phi_squared = phi.array().square();
            std::function<Vector(Vector)> f = [=](Vector x) {
                auto e_2_x = x.array().exp();
                return (e_2_x * (Delta_squared - phi_squared - v - e_2_x)).array() / (2 * (phi_squared + v + e_2_x).array().pow(2) - ((x - a) / gcem::pow(glicko2_system_rating_, 2)).array()).array();
            };
            std::function<double(double, int)> f_d = [=](double x, int i) {
                auto e_to_x = gcem::exp(x);
                return (e_to_x * (Delta_squared(i) - phi_squared(i) - v - e_to_x)) / (2 * gcem::pow(phi_squared(i) + v + e_to_x, 2)) - (x - a(i)) / (glicko2_system_rating_ * glicko2_system_rating_);
            };

            auto A = a;
            Vector B; B.setZero();

            for (int i = 0; i < n; i++) {
                if (gcem::pow(Delta(i), 2) > (v + gcem::pow(phi(i), 2))) {
                    B(i) = gcem::log(gcem::pow(Delta(i), 2) - gcem::pow(phi(i), 2) - v);
                } else {
                    int k = 1;
                    while (f_d(a(i) - k * glicko2_system_rating_, i) < 0) {
                        ++k;
                    }
                    B(i) = a(i) - k * glicko2_system_rating_;
                }
            }

            auto f_A = f(A);
            auto f_B = f(B);

            for (int i = 0; i < n; i++) {
                while(gcem::abs(B(i) - A(i)) > convergeance_tolerance) {
                    auto C = A(i) + (A(i) - B(i)) * f_A(i) / (f_B(i) - f_A(i));
                    auto f_C = f_d(C, i);
                    if (f_C * f_B(i) < 0) {
                        A(i) = B(i);
                        f_A(i) = f_B(i);
                    } else {
                        f_A(i) = f_A(i) / 2.0;
                    }

                    B(i) = C;
                    f_B(i) = f_C;
                }
            }

            glicko_volatility_ = (A / 2.0).array().exp();
            Vector phi_star = (phi_squared + glicko_volatility_.array().square()).sqrt();
            glicko_deviation_ = Eigen::rsqrt(1.0 / phi_star.array().square() + 1.0 / v);
            glicko_rating_ = ((mu + static_cast<Matrix>(glicko_deviation_.array().square()) * delta.transpose()) * 173.7178).array() + 1500.0;
            glicko_deviation_ *= 173.7178;
        }

    protected:
        Matrix hypothetical_elo_diff() {
            auto win_ratio_arr = win_ratio_matrix().array();
//            return 400.0 * (Eigen::log10(win_ratio_mat.array()) - Eigen::log10((1.0 - win_ratio_mat.array())));
            return 400.0 * (Eigen::log10(win_ratio_arr / (1.0 - win_ratio_arr)));
        }

        Matrix hypothetical_elo_diff() const {
            auto win_ratio_arr = win_ratio_matrix().array();
//            return 400.0 * (Eigen::log10(win_ratio_mat.array()) - Eigen::log10((1.0 - win_ratio_mat.array())));
            return 400.0 * (Eigen::log10(win_ratio_arr / (1.0 - win_ratio_arr)));
        }

        Matrix likelihood_ratio() {
            auto score_mat = score_matrix();
            auto var_s = (static_cast<Matrix>(win_loss_matrix_) + 0.25 * static_cast<Matrix>(draw_matrix_)).array() / static_cast<Vector>(num_games_).array() - score_mat.array().pow(2) / 0.5;
            Vector s0 = Eigen::inverse((1.0 + Eigen::pow(10, (-1.0 / (400.0 * static_cast<Vector>(elo_)).array()))));
            Matrix s;
            for (int i = 0; i < s0.size(); i++) {
                s.row(i) = s0;
            }
            return (s - s.transpose()) * static_cast<Matrix>((2 * score_mat - (s + s.transpose())).array() / (var_s * 2.0).array());
        }

        Matrix likelihood_ratio() const {
            auto score_mat = score_matrix();
            auto var_s = (static_cast<Matrix>(win_loss_matrix_) + 0.25 * static_cast<Matrix>(draw_matrix_)).array() / static_cast<Vector>(num_games_).array() - score_mat.array().pow(2) / 0.5;
            Vector s0 = Eigen::inverse((1.0 + Eigen::pow(10, (-1.0 / (400.0 * static_cast<Vector>(elo_)).array()))));
            Matrix s;
            for (int i = 0; i < s0.size(); i++) {
                s.row(i) = s0;
            }
            return (s - s.transpose()) * static_cast<Matrix>((2 * score_mat - (s + s.transpose())).array() / (var_s * 2.0).array());
        }

        Matrix sprt(const double alpha, const double beta) {
            auto llr = likelihood_ratio();
            auto la = gcem::log(beta / (1 - alpha));
            auto lb = gcem::log((1 - beta) / alpha);
            Matrix ret;
            for (int i = 0; i < llr.rows(); i++) {
                for (int j = 0; j < llr.cols(); j++) {
                    if (llr(i, j) > lb) {
                        ret(i, j) = 1;
                    } else if (llr(i, j) < la) {
                        ret(i, j) = -1;
                    }
                    ret(i, j) = 0;
                }
            }
            return ret;
        }

        Matrix sprt(const double alpha, const double beta) const {
            auto llr = likelihood_ratio();
            auto la = gcem::log(beta / (1 - alpha));
            auto lb = gcem::log((1 - beta) / alpha);
            Matrix ret;
            for (int i = 0; i < llr.rows(); i++) {
                for (int j = 0; j < llr.cols(); j++) {
                    if (llr(i, j) > lb) {
                        ret(i, j) = 1;
                    } else if (llr(i, j) < la) {
                        ret(i, j) = -1;
                    }
                    ret(i, j) = 0;
                }
            }
            return ret;
        }
    };

    /**
     * N.B. There are faster algorithms for counting inversions.
     * Tested.
     * computes the normalized Kendall tau distance of two rankings. Assumes the rankings are the same size and that
     * the first data set is sorted.
     * @tparam Iterator
     * @param first1
     * @param last1
     * @param first2
     * @return
     */
    template <class Iterator>
    double norm_kendall_tau_dist_sorted(Iterator first1, Iterator last1, Iterator first2) {
        assert(first1 != last1);
        if (first1 == last1) {
            throw std::logic_error("Attempting to find the normalized Kendall tau distance of empty vectors!");
        }

        auto size = std::distance(first1, last1);
        int inversion_count = 0;

        for (int i = 0; i < size - 1; i++) {
            for (int j = i + 1; j < size; j++) {
                if (*(first2 + j) < *(first2 + i)) {
                    inversion_count++;
                }
            }
        }

        return 2. * ((double)inversion_count / (size * (size - 1)));
    }
};

#endif //STATS_RANKING_HPP
