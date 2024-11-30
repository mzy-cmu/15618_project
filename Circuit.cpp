#include "Circuit.h"

Signal::Signal(string name, int id) : name(name), id(id) {}

Gate::Gate(string type, int output, vector<int> inputs) : type(type), output(output), inputs(inputs) {}

CircuitDAG::CircuitDAG() {}

// Add a signal to the circuit
int CircuitDAG::addSignal(const string &name) {
    if (signal_map.find(name) != signal_map.end()) {
        return signal_map[name];
    }
    int id = signals.size();
    signals.emplace_back(name, id);
    signal_map[name] = id;
    return id;
}

// Add a gate to the circuit
void CircuitDAG::addGate(const Gate &gate) {
    gates.push_back(gate);
    // Adjust size
    adj_list.resize(signals.size());
    in_degree.resize(signals.size(), 0);

    // Add edges from input signals to the gate's output
    for (size_t i = 0; i < gate.inputs.size(); i++) {
        int input = gate.inputs[i];
        adj_list[input].push_back(gate.output);
        in_degree[gate.output]++;
    }
}


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