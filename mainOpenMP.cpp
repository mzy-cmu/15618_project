#include "Circuit.h"
#include "Evaluate.h"
#include <chrono>
#include <omp.h>
using namespace std::chrono;

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
    auto parseStartTime = high_resolution_clock::now();
    // ********** Command Line Parsing **********
    // Read command line arguments
    int opt;
    int num_thread = 0;
    string circuit_filename, testcase_filename, mode;
    while ((opt = getopt(argc, argv, "f:t:m:n:")) != -1) {
        switch (opt) {
        case 'f':
            circuit_filename = string(optarg);
            break;
        case 't':
            testcase_filename = string(optarg);
            break;
        case 'm':
            mode = string(optarg);
            break;
        case 'n':
            num_thread = atoi(optarg);
        default:
            cerr << "Usage: " << argv[0] << " -f [circuit_filename] -t [testcase_filename] -m [S/T/F] -n [num_threads]\n";
        }
    }

    // Check if arguments were set
    if (circuit_filename.empty()) {
        cerr << "Error: -f [circuit_filename] is required.\n";
        cerr << "Usage: " << argv[0] << " -f [circuit_filename] -t [testcase_filename] -m [S/T/F] -n [num_threads]\n";
        return 1;
    }
    if (testcase_filename.empty()) {
        cerr << "Error: -t [testcase_filename] is required.\n";
        cerr << "Usage: " << argv[0] << " -f [circuit_filename] -t [testcase_filename] -m [S/T/F] -n [num_threads]\n";
        return 1;
    }
    if (mode.empty() || (mode != "S" && mode != "T" && mode != "F")) {
        cerr << "Error: -m [S/T/F] is required.\n";
        cerr << "Usage: " << argv[0] << " -f [circuit_filename] -t [testcase_filename] -m [S/T/F] -n [num_threads]\n";
        return 1;
    }
    if (num_thread == 0) {
        cerr << "Error: -n [number of threads] is required.\n";
        cerr << "Usage: " << argv[0] << " -f [circuit_filename] -t [testcase_filename] -m [S/T/F] -n [num_threads]\n";
        return 1;
    }

    // ********** Data Structure **********
    // Parsed data
    vector<int> inputs;                        // Input signal ids
    vector<int> outputs;                       // Output signal ids
    vector<string> signals;                    // Signal id -> name
    unordered_map<string, int> signal_map;     // Name -> signal id
    vector<GATETYPE> gate_type;                // Gate type
    vector<vector<int>> gate_input;            // Gate inputs
    vector<int> gate_input_size;               // Number of gate inputs
    vector<int> gate_input_startidx;           // Gate input array start index
    vector<vector<int>> dependent_signals;     // Siganl id -> dependent output signals
    vector<int> dependency_degree;             // Signal id -> In-degree of each signal

    // Identify dependency
    vector<bool> check_todo;                   // Mark proccessed signals
    vector<vector<int>> signals_todo;          // Parallelizable signals each round
    vector<int> signals_todo_size;             // Number of parallelizable signals each round
    vector<int> signals_todo_startidx;         // Parallelizable signal array start index

    // Testcase
    vector<vector<bool>> tests;                // Test cases
    vector<bool> values;                       // Signal id -> [parallel test case values]
    vector<vector<bool>> output_values;        // Correct output values for each testcase

    // ********** Benchmark Parsing **********
    size_t num_inputs, num_outputs, num_signals, num_testcase;
    int *num_gate_input = new int[1];
    try {
        // Parse circuit
        parseISCAS89(circuit_filename, inputs, outputs, signals, signal_map, dependent_signals, dependency_degree, gate_type, num_gate_input, gate_input, gate_input_size, gate_input_startidx);
        // Parse testcase
        num_inputs = inputs.size();
        num_outputs = outputs.size();
        num_signals = signals.size();
        num_testcase = static_cast<size_t>(parseTestcase(testcase_filename, tests, num_inputs));
    } catch (const runtime_error &e) {
        cerr << e.what() << endl;
        return 1;
    }

    // ********** Identify dependency **********
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

    // ********** Evaluate Correct Output **********
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
            evaluateGates_serial(values, gate_type, gate_input, signals_todo[i], num_signals);
        }
        // Save correct outputs
        for (size_t i = 0; i < outputs.size(); i++) {
            output_values[test_id].push_back(values[outputs[i]]);
        }
    }

    auto parseEndTime = high_resolution_clock::now();
    auto parseDuration = duration_cast<microseconds>(parseEndTime - parseStartTime);
    
    // ********** Evaluate Faulty Circuits **********
    auto evalStartTime = high_resolution_clock::now();
    bool *detected;
    if (mode == "S") { // Serial Implementation
        detected = new bool[num_testcase * num_signals];
        for (size_t test_id = 0; test_id < num_testcase; test_id++) {
            // Set testcase
            values.assign(signals.size(), false);
            for (size_t input_i = 0; input_i < num_inputs; input_i++) {
                values[inputs[input_i]] = tests[test_id][input_i];
            }
            // For each fault
            for (size_t fault_id = 0; fault_id < num_signals; fault_id++) {
                // Implement input fault
                if (fault_id < inputs.size()) values[fault_id] = !values[fault_id];

                for (int i = 1; i < depth; i++) {
                    evaluateGates_serial(values, gate_type, gate_input, signals_todo[i], fault_id);
                }

                // Evaluate detected
                detected[test_id * num_signals + fault_id] = false;
                for (size_t i = 0; i < outputs.size(); i++) {
                    detected[test_id * num_signals + fault_id] |= (values[outputs[i]] != output_values[test_id][i]);
                }

                // Revert input fault
                if (fault_id < inputs.size()) values[fault_id] = !values[fault_id];
            }
        }
    } else if (mode == "F") { // Testcase-Level Parallel Implementation, flatten testcase & fault
        detected = new bool[num_testcase * num_signals];
        // vector<vector<bool>> values_testcase(num_testcase, vector<bool>(num_signals, false));
        vector<vector<vector<bool>>> values_testcase_fault(num_testcase, vector<vector<bool>>(num_signals, vector<bool>(num_signals, false)));

        omp_set_num_threads(num_thread);
        #pragma omp parallel for
        for (size_t test_fault_id = 0; test_fault_id < num_testcase * num_signals; test_fault_id++) {
            size_t test_id = test_fault_id / num_signals;
            size_t fault_id = test_fault_id % num_signals;
            // Set testcase
            for (size_t input_i = 0; input_i < num_inputs; input_i++) {
                values_testcase_fault[test_id][fault_id][inputs[input_i]] = tests[test_id][input_i];
            }
            // For each fault
            // Implement input fault
            if (fault_id < inputs.size()) values_testcase_fault[test_id][fault_id][fault_id] = !values_testcase_fault[test_id][fault_id][fault_id];

            for (int i = 1; i < depth; i++) {
                evaluateGates_serial(values_testcase_fault[test_id][fault_id], gate_type, gate_input, signals_todo[i], fault_id);
            }

            // Evaluate detected
            detected[test_id * num_signals + fault_id] = false;
            for (size_t i = 0; i < outputs.size(); i++) {
                detected[test_id * num_signals + fault_id] |= (values_testcase_fault[test_id][fault_id][outputs[i]] != output_values[test_id][i]);
            }

            // Revert input fault
            if (fault_id < inputs.size()) values_testcase_fault[test_id][fault_id][fault_id] = !values_testcase_fault[test_id][fault_id][fault_id];
        }
    }
    else if (mode == "T") { // Testcase-Level Parallel Implementation
        detected = new bool[num_testcase * num_signals];
        // vector<vector<bool>> values_testcase(num_testcase, vector<bool>(num_signals, false));

        omp_set_num_threads(num_thread);
        #pragma omp parallel for
        for (size_t test_id = 0; test_id < num_testcase; test_id++) {
            // Set testcase
            vector<bool> values(num_signals, false);
            for (size_t input_i = 0; input_i < num_inputs; input_i++) {
                values[inputs[input_i]] = tests[test_id][input_i];
            }
            // For each fault
            vector<bool> detected_fault(num_signals, false);
            for (size_t fault_id = 0; fault_id < num_signals; fault_id++) {
                // Implement input fault
                if (fault_id < inputs.size()) values[fault_id] = !values[fault_id];

                for (int i = 1; i < depth; i++) {
                    evaluateGates_serial(values, gate_type, gate_input, signals_todo[i], fault_id);
                }

                // Evaluate detected
                for (size_t i = 0; i < outputs.size(); i++) {
                    detected_fault[fault_id] = detected_fault[fault_id] | (values[outputs[i]] != output_values[test_id][i]);
                }

                // Revert input fault
                if (fault_id < inputs.size()) values[fault_id] = !values[fault_id];

                // update global detected
                detected[test_id * num_signals + fault_id] = detected_fault[fault_id];
            }
        }
    }
    auto evalEndTime = high_resolution_clock::now();
    auto evalDuration = duration_cast<microseconds>(evalEndTime - evalStartTime);

    // ********** Output **********
    cout << "Parsing overhead: " << parseDuration.count() << " us" << endl;
    cout << "Evaluate: " << evalDuration.count() << " us" << endl;
    // Print number of faults detected by each testcase
    int num_detected = 0;
    for (size_t i = 0; i < num_testcase; i++) {
        for (size_t j = 0; j < num_signals; j++) {
            num_detected += detected[i * num_signals + j];
            // cout << detected[i * num_signals + j] << endl;
        }
        cout << i+1 << " " << num_detected << endl;
    }

    return 0;
}