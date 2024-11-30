#ifndef CIRCUIT_DAG_H
#define CIRCUIT_DAG_H

#include <vector>
#include <string>
using namespace std;

struct Gate {
    string type;                // Gate type (e.g., INPUT, AND, OR, NOT)
    vector<int> inputs;         // Input signal id
};

vector<int> outputs;                       // Output signal id
vector<string> signals;                    // Signal id -> name
vector<Gate> gates;                        // Signal id -> gate type (incl INPUT), inputs
vector<int> dependency_degree;             // Signal id -> In-degree of each signal
vector<vector<bool>> value;                // Signal id -> [parallel test case values]

int addSignal(const string &name);
void addGate(const Gate &gate);

#endif
