#ifndef CSVREADER_H
#define CSVREADER_H

#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <iostream>

class CSVReader {
public:
    CSVReader(const std::string& filename);

    // Returns data including header
    std::vector<std::vector<std::string>> getData() const;

    // Get header separately
    std::vector<std::string> getHeader() const;

    // Convert column to numeric if needed
    std::vector<double> getNumericColumn(size_t col_index) const;

private:
    std::vector<std::vector<std::string>> data;
    std::vector<std::string> header;

    void readFile(const std::string& filename);
};

#endif // CSVREADER_H
