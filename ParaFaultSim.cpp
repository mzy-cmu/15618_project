#include "Circuit.h"
#include "Evaluate.h"

using namespace std;

int main(int argc, char *argv[]) {
    // Read command line arguments
    int opt;
    string circuit_filename, testcase_filename;
    while ((opt = getopt(argc, argv, "f:t:")) != -1) {
        switch (opt) {
        case 'f':
            circuit_filename = string(optarg);
            break;
        case 't':
            testcase_filename = string(optarg);
            break;
        default:
            cerr << "Usage: " << argv[0] << " -f [circuit_filename] -t [testcase_filename]\n";
        }
    }

    // Check if filenames were set
    if (circuit_filename.empty()) {
        cerr << "Error: -f [circuit_filename] is required.\n";
        cerr << "Usage: " << argv[0] << " -f [circuit_filename] -t [testcase_filename]\n";
        return 1;
    }
    if (testcase_filename.empty()) {
        cerr << "Error: -t [testcase_filename] is required.\n";
        cerr << "Usage: " << argv[0] << " -f [circuit_filename] -t [testcase_filename]\n";
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
    vector<vector<bool>> tests;                // Test cases

    size_t num_inputs, num_testcase;
    try {
        // Parse circuit
        parseISCAS89(circuit_filename, inputs, outputs, signals, signal_map, gates, dependent_signals, dependency_degree);
        // Parse testcase
        num_inputs = inputs.size();
        num_testcase = static_cast<size_t>(parseTestcase(testcase_filename, tests, num_inputs));
    } catch (const runtime_error &e) {
        cerr << e.what() << endl;
        return 1;
    }

    // For each testcase
    size_t num_signals = signals.size();
    for (size_t test_id = 0; test_id < num_testcase; test_id++) {
        // Set testcase
        values.assign(signals.size(), false);

        for (size_t input_i = 0; input_i < num_inputs; input_i++) {
            values[inputs[input_i]] = tests[test_id][input_i];
            testcase
        }

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
                // TODO: add a serial version of evaluateGates
                evaluateGates(values, gates, signals_todo, num_signals);
            }
            batch_id++;
        }

        if (num_signals_accum != num_signals) {
            cerr << "Error: Proccessed signal count doesn't match\n";
            return 1;
        }
        
        // accumulate correct output values
        for (size_t i = 0; i < outputs.size(); i++) {
            output_values.push_back(values[outputs[i]]);
        }
    }

    return 0;
}