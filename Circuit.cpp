#include "Circuit.h"
#include <iostream>

// Add a signal to the circuit, return id
int addSignal(const string name,
              vector<string> &signals,
              unordered_map<string, int> &signal_map) {
    // Avoid redundent add
    if (signal_map.find(name) != signal_map.end()) {
        return signal_map[name];
    }
    int id = signals.size();
    signals.push_back(name);
    signal_map[name] = id;
    return id;
}

void parseGate(const string line,
               vector<string> &signals,
               unordered_map<string, int> &signal_map,
               vector<Gate> &gates,
               vector<vector<int>> &dependent_signals,
               vector<int> &dependency_degree) {
    string output, type, input_list;
    stringstream ss(line);

    // Output signal
    getline(ss, output, '=');
    output.erase(std::remove(output.begin(), output.end(), ' '), output.end());
    int output_id = addSignal(output, signals, signal_map);

    // Gate type and inputs
    getline(ss, type, '(');
    type.erase(std::remove(type.begin(), type.end(), ' '), type.end());
    getline(ss, input_list, ')');

    // Input signals
    vector<int> input_ids;
    stringstream inputStream(input_list);
    string input;
    while (getline(inputStream, input, ',')) {
        input.erase(std::remove(input.begin(), input.end(), ' '), input.end());
        input_ids.push_back(addSignal(input, signals, signal_map));
    }

    // Update gates, dependent_signals, dependency_degree
    gates.resize(signals.size());
    Gate gate = {type, input_ids};
    gates[output_id] = gate;

    dependency_degree.resize(signals.size(), 0);
    dependency_degree[output_id] = input_ids.size();
    dependent_signals.resize(signals.size());
    for (size_t i = 0; i < input_ids.size(); i++)
        dependent_signals[input_ids[i]].push_back(output_id);
}

void parseInputOutput(const string line,
                      const bool isOutput,
                      vector<int> &inputs,
                      vector<int> &outputs,
                      vector<string> &signals,
                      unordered_map<string, int> &signal_map,
                      vector<Gate> &gates,
                      vector<int> &dependency_degree) {
    string name;
    stringstream ss(line);
    getline(ss, name, '('); // Skip "INPUT" "OUTPUT"
    getline(ss, name, ')');

    int id = addSignal(name, signals, signal_map);
    if (!isOutput) {
        inputs.push_back(id);
        Gate gate = {"INPUT", {}};
        gates.push_back(gate);
        dependency_degree.push_back(0);
    }
    else
        outputs.push_back(id);
}

// Parse ISCAS89 and create circuit data structure
void parseISCAS89(const string filename,
                  vector<int> &inputs,
                  vector<int> &outputs,
                  vector<string> &signals,
                  unordered_map<string, int> &signal_map,
                  vector<Gate> &gates,
                  vector<vector<int>> &dependent_signals,
                  vector<int> &dependency_degree) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Error: Could not open file " + filename);
    }

    string line;
    while (getline(file, line)) {
        if (line.empty())
            continue;
        // Parse IO
        if (line.find("INPUT") == 0) {
            parseInputOutput(line, false, inputs, outputs, signals, signal_map, gates, dependency_degree);
        } else if (line.find("OUTPUT") == 0) {
            parseInputOutput(line, true, inputs, outputs, signals, signal_map, gates, dependency_degree);
        // Parse gates
        } else if (line.find("=") != string::npos) {
            parseGate(line, signals, signal_map, gates, dependent_signals, dependency_degree);
        }
    }

    file.close();
    return;
}

// Parse test case
int parseTestcase(const string &testcase_filename,
                  vector<vector<bool>> &tests,
                  size_t num_inputs) {
    ifstream file(testcase_filename);
    if (!file.is_open()) {
        cerr << "Error opening file: " << testcase_filename << endl;
        return 0;
    }

    string line;
    int test_id;
    int num_testcase = 0;

    while (getline(file, line)) {
        // Find the colon to split the test_id and binary string
        size_t colon_pos = line.find(':');
        if (colon_pos == string::npos) {
            cerr << "Invalid format in line: " << line << endl;
            continue;
        }

        // Extract test_id and binary string
        try {
            test_id = stoi(line.substr(0, colon_pos));
        } catch (const invalid_argument &) {
            cerr << "Invalid test id in line: " << line << endl;
            continue;
        }

        string binary_str = line.substr(colon_pos + 1);
        binary_str.erase(0, binary_str.find_first_not_of(" \t")); // Trim leading spaces

        // Validate the binary string length
        if (binary_str.size() != num_inputs) {
            cerr << "Error: Testcase " << test_id << " has wrong number of inputs" << endl;
            return 0;
        }

        // Convert binary string to boolean vector
        vector<bool> test_case;
        for (char ch : binary_str) {
            if (ch == '1') {
                test_case.push_back(true);
            } else if (ch == '0') {
                test_case.push_back(false);
            } else {
                cerr << "Invalid character in binary string: " << ch << endl;
                return 0;
            }
        }

        // Resize the tests vector to accommodate the test_id
        if (tests.size() <= static_cast<size_t>(test_id - 1)) {
            tests.resize(test_id);
        }

        // Store the test case
        tests[test_id - 1] = test_case;
        num_testcase++;
    }

    return num_testcase;
}

// Identify ready signals in this batch, resolve dependency
vector<int> popSignals(vector<bool> &check_todo,
                       vector<vector<int>> &dependent_signals,
                       vector<int> &dependency_degree) {
    vector<int> signals_todo;

    // Enqueue all signals with in-degree 0 that are not proccessed
    for (size_t signalIn = 0; signalIn < dependency_degree.size(); signalIn++) {
        if (!check_todo[signalIn] && dependency_degree[signalIn] == 0) {
            signals_todo.push_back(signalIn);
            check_todo[signalIn] = true;
        }
    }

    // Decrease in_degree of dependent signals
    for (size_t j = 0; j < signals_todo.size(); j++) {
        for (size_t i = 0; i < dependent_signals[signals_todo[j]].size(); i++) {
            int signalOut = dependent_signals[signals_todo[j]][i];
            dependency_degree[signalOut]--;
        }
    }

    return signals_todo;
}