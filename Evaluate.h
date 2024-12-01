#ifndef EVALUATE_H
#define EVALUATE_H

#include <iostream>
#include <vector>
#include <string>
#include <unistd.h>
#include "Circuit.h"

bool evaluateGate(vector<bool>& values, Gate gate);

void evaluateGates(vector<bool>& values, vector<Gate> gates, vector<int> signals_todo, int fault_id);

#endif