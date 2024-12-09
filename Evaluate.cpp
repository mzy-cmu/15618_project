#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include "Circuit.h"

bool evaluateGate(vector<bool>& values, Gate gate) {
    // value "true" is 1, "false" is 0
    bool and_gate = true; // AND & NAND gate result
    bool or_gate = false; // OR & NOR gate result
    bool xor_gate = values[gate.inputs[0]]; // XOR & XNOR gate result
    for (size_t i = 0; i < gate.inputs.size(); i++) {
        if (!values[gate.inputs[i]]) {
            and_gate = false; // if any value is zero, result is zero
        } else {
            or_gate = true; // if any value is one, result is zero
        }
        xor_gate = xor_gate ^ values[gate.inputs[i]];
    }
    if (gate.type == BUFF) {
        if (gate.inputs.size() != 1) {
            cerr << "Error: BUFF gate input size not 1\n";
            return 1;
        }
        return values[gate.inputs[0]];
    } else if (gate.type == NOT) {
        if (gate.inputs.size() != 1) {
            cerr << "Error: NOT gate input size not 1\n";
            return 1;
        }
        return !values[gate.inputs[0]];
    } else if (gate.type == AND) {
        return and_gate;
    } else if (gate.type == NAND) {
        return !and_gate;
    } else if (gate.type == OR) {
        return or_gate;
    } else if (gate.type == NOR) {
        return !or_gate;
    } else if (gate.type == XOR) {
        return xor_gate;
    } else if (gate.type == XNOR) {
        return !xor_gate;
    } else {
        cerr << "Error: Invalid gate type " << gate.type << "\n";
        return 1;
    }
}

void evaluateGates_serial(vector<bool>& values, vector<Gate> gates, vector<int> signals_todo, int fault_id) {
    for (size_t i = 0; i < signals_todo.size(); i++) {
        int gate_id = signals_todo[i];
        bool gate_value = evaluateGate(values, gates[gate_id]);
        if (gate_id == fault_id) {
            values[gate_id] = !gate_value;
        } else {
            values[gate_id] = gate_value;
        }
    }
}