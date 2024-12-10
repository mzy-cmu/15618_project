#ifndef EVALUATE_H
#define EVALUATE_H

#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include "Circuit.h"

bool evaluateGate(vector<bool>& values, GATETYPE gate_type, vector<int>& gate_input);

void evaluateGates_serial(vector<bool>& values, vector<GATETYPE>& gate_type, vector<vector<int>>& gate_input, vector<int> signals_todo, int fault_id);

#endif