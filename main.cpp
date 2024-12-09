#include "Circuit.h"
#include "Evaluate.h"

using namespace std;

bool *
ParaFaultSim(int numSignal, int numInput, GATETYPE *gateType, int numGateInput, int *gateInput, int *gateInputSize, int *gateInputStartIdx, int numTestcase, bool *testcase, int depth, int maxGatePara, int *gatePara, int *gateParaSize, int *gateParaStartIdx, int numOutput, int *outputId, bool *outputVal);

int* int_vec2arr(vector<vector<int>>& vec2D, int size) {
    // Allocate memory for the 1D array
    int* arr = new int[size];

    // Flatten the 2D vector into the 1D array
    int accum = 0;
    for (size_t i = 0; i < vec2D.size(); ++i) {
        for (size_t j = 0; j < vec2D[i].size(); ++j) {
            arr[accum + j] = vec2D[i][j];
        }
        accum += vec2D[i].size();
    }

    return arr;
}

bool* bool_vec2arr(vector<vector<bool>>& vec2D, int rows, int cols) {
    // Allocate memory for the 1D array
    bool* arr = new bool[rows * cols];

    // Flatten the 2D vector into the 1D array
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            arr[i * cols + j] = vec2D[i][j];
        }
    }

    return arr;
}

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
    vector<GATETYPE> gate_type;                // CUDA gate type
    vector<vector<int>> gate_input;            // CUDA gate inputs
    vector<int> gate_input_size;               // CUDA gate number of inputs
    vector<int> gate_input_startidx;           // CUDA gate input array start index
    vector<vector<int>> dependent_signals;     // Siganl id -> dependent output signals
    vector<int> dependency_degree;             // Signal id -> In-degree of each signal
    vector<bool> values;                       // Signal id -> [parallel test case values]
    vector<bool> check_todo;                   // Mark proccessed signals

    vector<vector<bool>> output_values;        // Correct output values for each testcase
    vector<vector<bool>> tests;                // Test cases

    vector<vector<int>> signals_todo;
    vector<int> signals_todo_size;
    vector<int> signals_todo_startidx;

    size_t num_inputs, num_outputs, num_signals, num_testcase;
    int *num_gate_input = new int[1];
    try {
        // Parse circuit
        parseISCAS89(circuit_filename, inputs, outputs, signals, signal_map, gates, dependent_signals, dependency_degree, gate_type, num_gate_input, gate_input, gate_input_size, gate_input_startidx);
        // Parse testcase
        num_inputs = inputs.size();
        num_outputs = outputs.size();
        num_signals = signals.size();
        num_testcase = static_cast<size_t>(parseTestcase(testcase_filename, tests, num_inputs));
        num_testcase = 1;
    } catch (const runtime_error &e) {
        cerr << e.what() << endl;
        return 1;
    }

    // Create todo lists
    check_todo.assign(signals.size(), false);
    size_t num_signals_accum = 0;
    int depth = 0;
    size_t max_signals_todo = 0;
    int startidx = 0;
    while (num_signals_accum < num_signals) {
        vector<int> signals_todo_new = popSignals(check_todo, dependent_signals, dependency_degree);
        signals_todo.push_back(signals_todo_new);

        size_t size = signals_todo_new.size();
        signals_todo_size.push_back(size);

        signals_todo_startidx.push_back(startidx);
        startidx += size;

        num_signals_accum += size;
        depth++;
        if (size > max_signals_todo) max_signals_todo = size;
    }

    if (num_signals_accum != num_signals) {
        cerr << "Error: Proccessed signal count doesn't match\n";
        return 1;
    }

    // Evaluate correct circuit
    output_values.resize(num_testcase);
    for (size_t test_id = 0; test_id < num_testcase; test_id++) {
        // Set testcase
        values.assign(signals.size(), false);
        for (size_t input_i = 0; input_i < num_inputs; input_i++) {
            values[inputs[input_i]] = tests[test_id][input_i];
        }
        // Evaluate gates
        for (int i = 1; i < depth; i++) {
            evaluateGates_serial(values, gates, signals_todo[i], num_signals);
        }
        // Save correct outputs
        for (size_t i = 0; i < outputs.size(); i++) {
            output_values[test_id].push_back(values[outputs[i]]);
        }
    }

    bool *testcases = bool_vec2arr(tests, num_testcase, num_inputs);
    int *signals_todo_arr = int_vec2arr(signals_todo, num_signals);
    bool *output_values_arr = bool_vec2arr(output_values, num_testcase, num_outputs);
    int *gate_input_arr = int_vec2arr(gate_input, *num_gate_input);

    for (size_t i = 0; i < signals.size(); i++) {
        cout << i << ":" << signals[i] << " ";
    }
    cout << endl;

    for (size_t i = 0; i < num_signals; i++) {
        cout << i << ":" << signals_todo_arr[i] << " ";
    }
    cout << endl;

    for (size_t i = 0; i < num_testcase * num_outputs; i++) {
        cout << output_values_arr[i];
    }
    cout << endl;

    // Evaluate faulty circuits
    bool *detected;
    detected = ParaFaultSim(num_signals, num_inputs, gate_type.data(), *num_gate_input, gate_input_arr, gate_input_size.data(), gate_input_startidx.data(), num_testcase, testcases, depth, max_signals_todo, signals_todo_arr, signals_todo_size.data(), signals_todo_startidx.data(), num_outputs, outputs.data(), output_values_arr);

    for (size_t i = 0; i < num_testcase; i++) {
        for (size_t j = 0; j < num_signals; j++) {
            cout << detected[i * num_signals + j];
        }
        cout << endl;
    }

    return 0;
}