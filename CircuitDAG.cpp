#include "CircuitDAG.h"

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
