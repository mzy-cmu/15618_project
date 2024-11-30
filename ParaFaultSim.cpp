#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include <unistd.h>
#include "Circuit.h"

int main(int argc, char *argv[]) {
    // Read command line arguments
    int opt;
    string filename;
    while ((opt = getopt(argc, argv, "f:")) != -1) {
        switch (opt) {
        case 'f':
            filename = string(optarg);
            break;
        default:
            std::cerr << "Usage: " << argv[0] << " -f [input_filename]\n";
        }
    }

    // Check if filename was set
    if (filename.empty()) {
        std::cerr << "Error: -f [input_filename] is required.\n";
        std::cerr << "Usage: " << argv[0] << " -f [input_filename]\n";
        return 1;
    }

    // Data structures
    vector<int> outputs;                       // Output signal id
    vector<string> signals;                    // Signal id -> name
    vector<Gate> gates;                        // Signal id -> gate type (incl INPUT), inputs
    vector<int> dependency_degree;             // Signal id -> In-degree of each signal
    vector<vector<bool>> values;               // Signal id -> [parallel test case values]

    // Parse and init
    try {
        CircuitDAG DAG = parseISCAS89(filename);
        vector<int> topo_order = topologicalSort(DAG);
        size_t size = topo_order.size();
        for (size_t i = 0; i < size; i++) {
            cout << topo_order[i] << " " << DAG.signals[topo_order[i]].name << endl;
        }
    } catch (const runtime_error &e) {
        cerr << e.what() << endl;
        return 1;
    }

    return 0;
}