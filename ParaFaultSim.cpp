#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include <queue>
#include "Circuit.h"

using namespace std;

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
        cerr << "Error: -f [input_filename] is required.\n";
        cerr << "Usage: " << argv[0] << " -f [input_filename]\n";
        return 1;
    }

    // Data structures
    vector<int> outputs;                       // Output signal id
    vector<string> signals;                    // Signal id -> name
    unordered_map<string, int> signal_map;     // Name -> signal id
    vector<Gate> gates;                        // Signal id -> gate type (incl INPUT), input ids
    vector<vector<int>> dependent_signals;      // Siganl id -> dependent output signals
    vector<int> dependency_degree;             // Signal id -> In-degree of each signal
    vector<vector<bool>> values;               // Signal id -> [parallel test case values]
    vector<bool> check_todo;                   // Mark proccessed signals

    // Parse and init
    try {
        parseISCAS89(filename, outputs, signals, signal_map, gates, dependent_signals, dependency_degree);
        check_todo.resize(signals.size(), false);
    } catch (const runtime_error &e) {
        cerr << e.what() << endl;
        return 1;
    }

    // Batch process
    size_t num_signals = signals.size();
    size_t num_signals_accum = 0;
    int batch_id = 0;
    while (num_signals_accum < num_signals) {
        vector<int> signals_todo = popSignals(check_todo, dependent_signals, dependency_degree);
        size_t size = signals_todo.size();
        num_signals_accum += size;

        batch_id++;
        cout << "Batch " << batch_id << ", "<< size << " wires: ";
        for (size_t i = 0; i < size; i++) {
            cout << signals_todo[i] << "(" << signals[signals_todo[i]] << ") ";
        }
        cout << "\n";
    }

    if (num_signals_accum != num_signals) {
        cerr << "Error: Proccessed signal count doesn't match\n";
    }

    return 0;
}