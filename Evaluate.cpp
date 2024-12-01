#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include "Circuit.h"

bool evaluateGate(vector<bool>& values, Gate gate) {
    if (gate.type == "BUFF") {
        if (gate.inputs.size() != 1) {
            cerr << "Error: BUFF gate input size not 1\n";
            return 1;
        }
        return values[gate.inputs[0]];
    } else if (gate.type == "NOT") {
        if (gate.inputs.size() != 1) {
            cerr << "Error: NOT gate input size not 1\n";
            return 1;
        }
        return !values[gate.inputs[0]];
    } else if (gate.type == "AND") {
        for (size_t i = 0; i < gate.inputs.size(); i++) {
            if (!values[gate.inputs[i]]) {
                return false;
            }
        }
        return true;
    } else if (gate.type == "NAND") {
        for (size_t i = 0; i < gate.inputs.size(); i++) {
            if (!values[gate.inputs[i]]) {
                return true;
            }
        }
        return false;
    } else if (gate.type == "OR") {
        for (size_t i = 0; i < gate.inputs.size(); i++) {
            if (values[gate.inputs[i]]) {
                return true;
            }
        }
        return false;
    } else if (gate.type == "NOR") {
        for (size_t i = 0; i < gate.inputs.size(); i++) {
            if (values[gate.inputs[i]]) {
                return false;
            }
        }
        return true;
    } else if (gate.type == "XOR") {
        bool parity_flag = values[gate.inputs[0]];
        for (size_t i = 1; i < gate.inputs.size(); i++) {
            parity_flag = parity_flag ^ values[gate.inputs[0]];
        }
        return parity_flag;
    } else if (gate.type == "XNOR") {
        bool parity_flag = values[gate.inputs[0]];
        for (size_t i = 1; i < gate.inputs.size(); i++) {
            parity_flag = parity_flag ^ values[gate.inputs[0]];
        }
        return !parity_flag;
    } else {
        cerr << "Error: Invalid gate type " << gate.type << "\n";
        return 1;
    }
}

void evaluateGates(vector<bool>& values, vector<Gate> gates, vector<int> signals_todo, int fault_id) {
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