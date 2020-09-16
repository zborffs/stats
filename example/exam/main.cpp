//
// Created by Zach Bortoff on 2020-09-15.
//

#include <iostream>
#include <filesystem>
#include <vector>
#include <string>
#include <fstream>

/**
 * reads a file given a path to the file and returns a vector of strings which are each line of the file
 * @param file_name the path to the file and the file name
 * @return a vectir if strubgs for each line in the file
 */
std::vector<std::string> read_file(const std::string& file_name) {
    std::vector<std::string> lines;

    /// sandwich in a try block to catch an I/O errors
    try {
        /// open the file in a fstream
        std::fstream in(file_name);

        /// catch any exceptions that may occur
        in.exceptions(std::ifstream::badbit);
        if(!in.is_open()) {
            std::string err("Failed to open file: ");
            err += file_name;
            throw std::runtime_error(err);
        }

        /// for each line in the fstream, push them on the return vector
        std::string line;
        for(int i = 1; std::getline(in, line); i++) {
            lines.push_back(line);
        }


        /// if you catch a badbit error, throw a runtime error
        if(in.bad()) {
            throw std::runtime_error("Runtime error in read_file(const std::string&): Badbit file.");
        }

        /// close the file
        in.close();

    } catch(const std::exception& e) {
        throw;
    }

    /// return the vector<string>
    return lines;
}

/**
 * splits an input string by an input delimiter
 * @param str input string
 * @param delim delimeter
 * @return a std::vector<std::string> whose elements are subsets of the input string as delimited by the delimter
 */
std::vector<std::string> split(const std::string& str, const char delim) {
    std::vector<std::string> ret;

    /// set the first and second iterators as the constant iterators at the beginning of the input string
    std::string::const_iterator first  = str.cbegin();
    std::string::const_iterator second = str.cbegin();

    /// for each character in the string, if the second iterator is at a delimter, and then put the substring
    /// sandwitched by the two iterators in the vector, otherwise increment the second iterator
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

struct ExamData {
    bool is_male;
    int ethnic;
    std::string parent_edu;
    bool free_lunch;
    bool prepared;
    int math_score;
    int reading_score;
    int writing_score;

    friend std::ostream& operator<<(std::ostream& os, ExamData& data) {
        std::cout << "ExamData:" << std::endl;
        os << std::boolalpha;
        std::cout << "- is_male: " << data.is_male << std::endl;
        std::cout << "  ethnic: " << data.ethnic;
        os << std::noboolalpha;
        return os;
    }
};

int main(int argc, char** argv) {
    std::cout << "Hello World!" << std::endl;

    /// Path to file with NIST data
    auto current_path = std::filesystem::current_path();
    std::string exam_data = current_path.string() + "/../../../example/exam/data/exams_10.csv";
    std::vector<std::string> lines;

    /// reads all the lines of the input data file
    try {
        lines = read_file(exam_data);
    } catch (...) {
        std::cout << "Error was caught: Exiting!" << std::endl;
    }

    std::vector<std::string> split_line;
    std::vector<ExamData> exam_data_vector;

    /// initializes two vectors with the data from the NIST file
    for (int i = 1; i < lines.size(); i++) {
        split_line = split(lines[i], ',');

        ExamData data{};
        for (int j = 0; j < split_line.size(); j++) {
            switch(j) {
                case 0: data.is_male = (split_line[j] == "\"male\""); break;
                case 1: data.ethnic = split_line[1][7] - 'A'; break;
            }
        }
        exam_data_vector.push_back(data);
    }

    for (auto & data : exam_data_vector) {
        std::cout << data << std::endl;
    }

    return 0;
}