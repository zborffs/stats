//
// Created by Zach Bortoff on 2020-03-09.
//

/// External Library Includes
#include <gtest/gtest.h>

/// Internal Library Includes
#include "chess.hpp"

/// This is arbitrary, but std::numeric_limits<double>::min() is like 120 orders of magnitude off.
const double ERROR = gcem::pow(10, -10);

class ChessTester : public ::testing::Test {
protected:
    std::streambuf* sbuf_;
    std::stringstream buffer_{};
    stats::Player prometheus_{"Prometheus"};
    stats::Player magnus_carlsen_{"Magnus Carlsen", 2862};
    stats::Player stockfish_{"Stockfish", 3500};
    stats::Player hikaru_nakamura_{"Hikaru Nakamura", 2736};

    stats::Player p1{"Player 1", 1500};
    stats::Player p2{"Player 2", 1400};
    stats::Player p3{"Player 3", 1550};
    stats::Player p4{"Player 4", 1700};

    ChessTester() : sbuf_{nullptr} {

    }

    ~ChessTester() override = default;

protected:
    virtual void SetUp() override {
        sbuf_ = std::cout.rdbuf();
        std::cout.rdbuf(buffer_.rdbuf());
    }

    void TearDown() override {
        std::cout.rdbuf(sbuf_);
        sbuf_ = nullptr;
    }
};

TEST_F(ChessTester, NumGames) {

}

TEST_F(ChessTester, ScoreDiff) {

}

TEST_F(ChessTester, Score) {

}

TEST_F(ChessTester, WinRatio) {

}

TEST_F(ChessTester, ExpectedWinRatio) {

}

TEST_F(ChessTester, LOS) {

}

TEST_F(ChessTester, ELODiff) {

}

TEST_F(ChessTester, LLR) {

}

TEST_F(ChessTester, SPRT) {

}

TEST_F(ChessTester, UpdateGlickoRating) {
    std::vector<stats::Player> players{p1, p2, p3, p4};

    stats::Tournament t = stats::Tournament(players, 1);

    stats::Match m1(p1, p2, 0, 0, 0, 0, 1);
    stats::Match m2(p1, p3, 0, 0, 0, 0, -1);
    stats::Match m3(p1, p4, 0, 0, 0, 0, -1);
    t.add_match(m1); t.add_match(m2); t.add_match(m3);
    t.update_glicko_ratings();
}

TEST_F(ChessTester, SimulatedTournament) {

}

TEST_F(ChessTester, PlayerOutput) {
    auto f = [=](const stats::Player& p) {
        return "Player:\n  name: " + p.name + "\n  elo: " + (std::to_string(p.elo_rating))+ "\n  glicko2: " + (std::to_string(p.glicko_rating - 2 * p.glicko_deviation)) + "-" + (std::to_string(p.glicko_rating + 2 * p.glicko_deviation)) + " (0.95, " + std::to_string(p.glicko_volatility) + ")\n";
    };

    std::cout << prometheus_;
    std::string actual{buffer_.str()};
    EXPECT_EQ(f(prometheus_), actual);

    std::cout << std::endl;
    std::cout.clear();
    buffer_.str("");

    std::cout << magnus_carlsen_;
    actual = std::string{buffer_.str()};
    EXPECT_EQ(f(magnus_carlsen_), actual);

    std::cout << std::endl;
    std::cout.clear();
    buffer_.str("");

    std::cout << hikaru_nakamura_;
    actual = std::string{buffer_.str()};
    EXPECT_EQ(f(hikaru_nakamura_), actual);

    std::cout << std::endl;
    std::cout.clear();
    buffer_.str("");

    std::cout << stockfish_;
    actual = std::string{buffer_.str()};
    EXPECT_EQ(f(stockfish_), actual);
}

TEST_F(ChessTester, MatchOutput) {

}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
