#include "CSVReader.h"

CSVReader::CSVReader(const std::string& filename) {
    readFile(filename);
}

void CSVReader::readFile(const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        return;
    }

    std::string line;
    bool firstLine = true;

    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::vector<std::string> row;
        std::string cell;

        while (std::getline(ss, cell, ',')) {
            row.push_back(cell);
        }

        if (firstLine) {
            header = row;
            firstLine = false;
        } else {
            data.push_back(row);
        }
    }
    file.close();
}

std::vector<std::vector<std::string>> CSVReader::getData() const {
    return data;
}

std::vector<std::string> CSVReader::getHeader() const {
    return header;
}

std::vector<double> CSVReader::getNumericColumn(size_t col_index) const {
    std::vector<double> numericColumn;
    for (const auto& row : data) {
        if (col_index < row.size()) {
            try {
                numericColumn.push_back(std::stod(row[col_index]));
            } catch (const std::invalid_argument&) {
                numericColumn.push_back(0.0); // or NaN, as fallback
            }
        } else {
            numericColumn.push_back(0.0);
        }
    }
    return numericColumn;
}
