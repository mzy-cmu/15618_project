// evaluate
// vector signal id
// vector bool 2D: 1D: signal id, 2D: testcase value
// Gate -> type, 

// 1: 2 3 2: 3 3:
// 1:     2: 1 3: 1 2
// 1: 0 2: 1 3: 2

// good circuit evaluate -> fault insertion for all signals then evaluate -> change testcase

for testcase
    for signal_id_list
        evaluateGate(signal_id)

vector<int> evaluateGates(vector<Gate> gates, vector<int> process_ids, int fault_id) {
    for(int i = 0; i < process_ids.size(); i++) {
        int gate_id = process_ids[i];
        bool gate_value = evaluateGate(gates[gate_id]);
        if (gate_id == fault_id) {
            value[gate_id] = !gate_value;
        } else {
            value[gate_id] = gate_value;
        }
    }

}

vector<int> evaluateGate(Gate gate) {
    switch(gate.type) {
        case "AND": 
            for (int i = 0; i < gate.inputs.size(); i++) {
                if ()
            }
    }
}