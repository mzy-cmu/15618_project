#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include "Circuit.h"

bool evaluateGate(vector<bool>& values, GATETYPE gate_type, vector<int>& gate_input) {
    // value "true" is 1, "false" is 0
    bool and_gate = true; // AND & NAND gate result
    bool or_gate = false; // OR & NOR gate result
    bool xor_gate = values[gate_input[0]]; // XOR & XNOR gate result
    for (size_t i = 0; i < gate_input.size(); i++) {
        if (!values[gate_input[i]]) {
            and_gate = false; // if any value is zero, result is zero
        } else {
            or_gate = true; // if any value is one, result is zero
        }
        if (i != 0) xor_gate = xor_gate ^ values[gate_input[i]];
    }
    if (gate_type == BUFF) {
        return values[gate_input[0]];
    } else if (gate_type == NOT) {
        return !values[gate_input[0]];
    } else if (gate_type == AND) {
        return and_gate;
    } else if (gate_type == NAND) {
        return !and_gate;
    } else if (gate_type == OR) {
        return or_gate;
    } else if (gate_type == NOR) {
        return !or_gate;
    } else if (gate_type == XOR) {
        return xor_gate;
    } else if (gate_type == XNOR) {
        return !xor_gate;
    } else {
        cerr << "Error: Invalid gate type " << gate_type << "\n";
        return 1;
    }
}

void evaluateGates_serial(vector<bool>& values, vector<GATETYPE>& gate_type, vector<vector<int>>& gate_input, vector<int> signals_todo, int fault_id) {
    for (size_t i = 0; i < signals_todo.size(); i++) {
        int gate_id = signals_todo[i];
        bool gate_value = evaluateGate(values, gate_type[gate_id], gate_input[gate_id]);
        if (gate_id == fault_id) {
            values[gate_id] = !gate_value;
        } else {
            values[gate_id] = gate_value;
        }
    }
}