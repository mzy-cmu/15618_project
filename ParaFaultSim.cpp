#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include "Circuit.h"
#include "Evaluate.h"

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
    vector<int> inputs;                        // Input signal ids
    vector<int> outputs;                       // Output signal ids
    vector<string> signals;                    // Signal id -> name
    unordered_map<string, int> signal_map;     // Name -> signal id
    vector<Gate> gates;                        // Signal id -> gate type (incl INPUT), input ids
    vector<vector<int>> dependent_signals;     // Siganl id -> dependent output signals
    vector<int> dependency_degree;             // Signal id -> In-degree of each signal
    vector<bool> values;                       // Signal id -> [parallel test case values]
    vector<bool> check_todo;                   // Mark proccessed signals

    vector<bool> output_values;                // Correct output values
    vector<vector<int>> testcase_faults;       // Testcase -> [stuck-at faults that can be detected, in signal id]

    // Parse and init
    try {
        parseISCAS89(filename, inputs, outputs, signals, signal_map, gates, dependent_signals, dependency_degree);
    } catch (const runtime_error &e) {
        cerr << e.what() << endl;
        return 1;
    }

    size_t num_signals = signals.size();

    // For each testcase
    size_t num_inputs = inputs.size();
    size_t num_testcase = 1 << inputs.size();
    for (size_t test_id = 0; test_id < 1; test_id++) {
        // Set testcase
        values.assign(signals.size(), false);

        cout << "\n*********Testcase " << test_id << ": ";
        for (size_t input_i = 0; input_i < num_inputs; input_i++) {
            values[inputs[input_i]] = bool((test_id >> input_i) & 1);
            cout << inputs[input_i] << "(" << signals[inputs[input_i]] << "):" << values[inputs[input_i]] << " ";
        }
        cout << "*********\n";

        // Evaluate good circuit
        check_todo.assign(signals.size(), false);
        vector<int> dependency_degree_work = dependency_degree;
        size_t num_signals_accum = 0;
        int batch_id = 0;
        while (num_signals_accum < num_signals) {
            vector<int> signals_todo = popSignals(check_todo, dependent_signals, dependency_degree_work);
            size_t size = signals_todo.size();
            num_signals_accum += size;
            
            // Evaluate gates in batch
            if (batch_id) {
                cout << "Batch " << batch_id << ", "<< size << " wires: ";
                for (size_t i = 0; i < size; i++) {
                    cout << signals_todo[i] << "(" << signals[signals_todo[i]] << ") ";
                }
                cout << "\n";

                evaluateGates(values, gates, signals_todo, num_signals);
            }
            batch_id++;
        }

        if (num_signals_accum != num_signals) {
            cerr << "Error: Proccessed signal count doesn't match\n";
            return 1;
        }

        cout << "Output: ";
        for (size_t i = 0; i < outputs.size(); i++) {
            output_values.push_back(values[outputs[i]]);
            cout << outputs[i] << "(" << signals[outputs[i]] << "):" << values[outputs[i]] << " ";
        }
        cout << "\n";

        // For each fault
        for (size_t fault_id = 0; fault_id < num_signals; fault_id++) {
            // Evaluate faulty circuit, same as above
            cout << "Fault: "<< fault_id << "(" << signals[fault_id] << ")\n";

            // Implement input fault
            if (fault_id < inputs.size())
                values[fault_id] = !values[fault_id];

            check_todo.assign(check_todo.size(), false);
            dependency_degree_work = dependency_degree;
            num_signals_accum = 0;
            batch_id = 0;
            while (num_signals_accum < num_signals) {
                vector<int> signals_todo = popSignals(check_todo, dependent_signals, dependency_degree_work);
                size_t size = signals_todo.size();
                num_signals_accum += size;

                if (batch_id) {
                    evaluateGates(values, gates, signals_todo, fault_id);
                }
                batch_id++;
            }

            // Revert input fault
            if (fault_id < inputs.size())
                values[fault_id] = !values[fault_id];

            bool diff = false;
            cout << "Output: ";
            for (size_t i = 0; i < signals.size(); i++) {
                // cout << outputs[i] << "(" << signals[outputs[i]] << "):" << values[outputs[i]] << " ";
                // if (values[outputs[i]] != output_values[i]) {
                //     diff = true;
                // }
                cout << i << "(" << signals[i] << "):" << values[i] << " ";
            }
            if (diff) {
                cout << " *";
            }
            cout << "\n";
        }
    }

    return 0;
}