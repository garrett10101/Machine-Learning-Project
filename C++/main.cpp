#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>

using namespace std;

vector<vector<double>> readCSV(const string &filename) {
    vector<vector<double>> data;
    ifstream file(filename);

    if (!file.is_open()) {
        cerr << "Error opening file: " << filename << endl;
        return data;
    }

    string line;
    bool header = true;

    while (getline(file, line)) {
        if (header) { header = false; continue; } // skip header

        vector<double> row;
        stringstream ss(line);
        string cell;

        while (getline(ss, cell, ',')) {
            row.push_back(stod(cell));
        }

        data.push_back(row);
    }

    file.close();
    return data;
}

int main() {
    vector<vector<double>> dataset = readCSV("cleaned_output_data.csv");

    // Print first few rows
    cout << "First 5 rows of dataset:" << endl;
    for (int i = 0; i < min(5, (int)dataset.size()); ++i) {
        for (double val : dataset[i]) {
            cout << val << " ";
        }
        cout << endl;
    }

    return 0;
}
