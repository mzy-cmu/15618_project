#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <queue>
#include <algorithm>
#include "CircuitDAG.h"

void parseInputOutput(const string &line, CircuitDAG &DAG, bool isInput) {
    string token;
    stringstream ss(line);
    getline(ss, token, '('); // Skip "INPUT" or "OUTPUT"

    while (getline(ss, token, ',')) {
        // Remove ) and space
        token.erase(std::remove(token.begin(), token.end(), ')'), token.end());
        token.erase(std::remove(token.begin(), token.end(), ' '), token.end());
        int id = DAG.addSignal(token);
        if (isInput) {
            DAG.inputs.push_back(id);
        } else {
            DAG.outputs.push_back(id);
        }
    }
}

void parseGate(const string &line, CircuitDAG &DAG) {
    string output, type, inputList;
    stringstream ss(line);

    // Output signal
    getline(ss, output, '=');
    output.erase(std::remove(output.begin(), output.end(), ' '), output.end());
    int outputID = DAG.addSignal(output);

    // Gate type and inputs
    getline(ss, type, '(');
    type.erase(std::remove(type.begin(), type.end(), ' '), type.end());
    getline(ss, inputList, ')');

    // Input signals
    vector<int> inputIDs;
    stringstream inputStream(inputList);
    string input;
    while (getline(inputStream, input, ',')) {
        input.erase(std::remove(input.begin(), input.end(), ' '), input.end());
        inputIDs.push_back(DAG.addSignal(input));
    }

    // Add the gate to the DAG
    Gate gate(type, outputID, inputIDs);
    DAG.addGate(gate);
}

// Parse ISCAS89 and create circuit data structure
CircuitDAG parseISCAS89(const string &filename) {
    ifstream file(filename);
    if (!file.is_open()) {
        throw runtime_error("Error: Could not open file " + filename);
    }

    CircuitDAG DAG;
    string line;

    while (getline(file, line)) {
        if (line.empty())
            continue;
        // Parse IO
        if (line.find("INPUT") == 0) {
            parseInputOutput(line, DAG, true);
        } else if (line.find("OUTPUT") == 0) {
            parseInputOutput(line, DAG, false);
        // Parse gates
        } else if (line.find("=") != string::npos) {
            parseGate(line, DAG);
        }
    }

    file.close();
    return DAG;
}

// Identify ready signals to be processed next round and resolve dependency
vector<int> topologicalSort(CircuitDAG &DAG) {
    vector<int> topo_order;
    queue<int> q;

    // Enqueue all signals with in-degree 0
    for (size_t i = 0; i < DAG.in_degree.size(); i++) {
        if (DAG.in_degree[i] == 0) {
            q.push(i);
        }
    }

    // Process all signals
    while (!q.empty()) {
        int signalIn = q.front();
        q.pop();
        topo_order.push_back(signalIn);

        // Decrease in_degree of dependent signals
        for (size_t i = 0; i < DAG.adj_list[signalIn].size(); i++) {
            int signalOut = DAG.adj_list[signalIn][i];
            DAG.in_degree[signalOut]--;
            // Refill queue
            if (DAG.in_degree[signalOut] == 0) {
                q.push(signalOut);
            }
        }
    }

    // Check for cycles
    if (topo_order.size() != DAG.signals.size()) {
        throw runtime_error("Error: The circuit contains a cycle!");
    }

    return topo_order;
}

int main() {
    const string filename = "Benchmarks/s27.bench";

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