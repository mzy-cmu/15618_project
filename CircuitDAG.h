#ifndef CIRCUIT_DAG_H
#define CIRCUIT_DAG_H

#include <vector>
#include <unordered_map>
#include <string>
using namespace std;

class Signal {
public:
    string name;
    int id;

    Signal(string name, int id);
};

class Gate {
public:
    string type;                // Gate type (e.g., AND, OR, NOT)
    int output;                 // Output signal ID
    vector<int> inputs;         // Input signal IDs

    Gate(string type, int output, vector<int> inputs);
};

// Represent the circuit as a Directed Acyclic Graph
class CircuitDAG {
public:
    vector<Signal> signals;
    vector<int> inputs;
    vector<int> outputs;
    vector<Gate> gates;
    unordered_map<string, int> signal_map;
    vector<vector<int>> adj_list;              // Mark IO signal dependency: gate input:[outputs]
    vector<int> in_degree;                     // In-degree of each gate

    CircuitDAG();
    int addSignal(const string &name);
    void addGate(const Gate &gate);
};

#endif
