#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "Circuit.h"

__device__
bool evaluateGate(bool *values, Gate gate) {
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
    if (gate.type == "BUFF") {
        return values[gate.inputs[0]];
    } else if (gate.type == "NOT") {
        return !values[gate.inputs[0]];
    } else if (gate.type == "AND") {
        return and_gate;
    } else if (gate.type == "NAND") {
        return !and_gate;
    } else if (gate.type == "OR") {
        return or_gate;
    } else if (gate.type == "NOR") {
        return !or_gate;
    } else if (gate.type == "XOR") {
        return xor_gate;
    } else if (gate.type == "XNOR") {
        return !xor_gate;
    }
}

__global__ void
evaluateGates_kernel(Gate *gates, bool *testcase,
                     int depth, int *gatePara, int *gateParaSize, int *gateParaStartIdx,
                     int numOutput, int *outputId, bool *outputVal, bool *detected) {
    int numSignal = blockDim.x;
    int numTestcase = blockDim.y;
    int gateIdx = threadIdx.x; 
    int testcaseIdx = blockIdx.y;
    int faultIdx = blockIdx.x;

    extern __shared__ bool values[];
    for (int i = 0; i < depth; i++) {
        if (gateIdx < gateParaSize[i]) {
            int gateId = gatePara[gateParaStartIdx[i] + gateIdx];
            if (i == 0) {
                // Assign testcase to input values
                values[gateId] = testcase[testcaseIdx * gateParaSize[0] + gateIdx];
                if (gateId == faultIdx) {
                    values[gateId] = !values[gateId];
                }
            }
            // Evaluate gates
            else {
                bool gateValue = evaluateGate(values, gates[gateId]);
                values[gateId] = gateValue;
                if (gateId == faultIdx) {
                    values[gateId] = !gateValue;
                }
            }
            __syncthreads();
        }
    }
    // Save output values
    if (gateIdx == 0) {
        for (int i = 0; i < numOutput; i++) {
            detected[testcaseIdx * numSignal + faultIdx] = values[outputId[i]] != outputVal[testcaseIdx * numOutput + i];
        }
    }
}

bool *
ParaFaultSim(int numSignal, int numInput, Gate *gates, int numTestcase, bool *testcase, int depth, int maxGatePara, int *gatePara, int *gateParaSize, int *gateParaStartIdx, int numOutput, int *outputId, bool *outputVal) {

    Gate *device_gates; // 1D gates[signalID]
    bool *device_testcase; // 2D test[testID][inputID]
    int *device_gatePara; // 1D gatePara[signalID]
    int *device_gateParaSize; // 1D gateParaSize[depth]
    int *device_gateParaStartIdx; // 1D gateParaStartIdx[depth]
    int *device_outputId; // 1D signalID[outputID]
    bool *device_outputVal; // correct output values, 2D outputVal[testID][outputID]
    bool *device_detected; // 2D detected[testID][faultID]

    // Allocate device memory buffers on the GPU using cudaMalloc
    cudaMalloc(&device_gates, sizeof(Gate) * numSignal);
    cudaMalloc(&device_testcase, sizeof(bool) * numTestcase * numInput);
    cudaMalloc(&device_gatePara, sizeof(int) * numSignal);
    cudaMalloc(&device_gateParaSize, sizeof(int) * depth);
    cudaMalloc(&device_gateParaStartIdx, sizeof(int) * depth);
    cudaMalloc(&device_outputId, sizeof(int) * numOutput);
    cudaMalloc(&device_outputVal, sizeof(int) * numTestcase * numOutput);
    cudaMalloc(&device_detected, sizeof(bool) * numTestcase * numSignal);

    // Start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    // Copy input arrays to the GPU using cudaMemcpy
    cudaMemcpy(device_gates, gates, sizeof(Gate) * numSignal, cudaMemcpyHostToDevice);
    cudaMemcpy(device_testcase, testcase, sizeof(bool) * numTestcase * numInput, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gatePara, gatePara, sizeof(int) * numSignal, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gateParaSize, gateParaSize, sizeof(int) * depth, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gateParaStartIdx, gateParaStartIdx, sizeof(int) * depth, cudaMemcpyHostToDevice);
    cudaMemcpy(device_outputId, outputId, sizeof(int) * numOutput, cudaMemcpyHostToDevice);
    cudaMemcpy(device_outputVal, outputVal, sizeof(int) * numTestcase * numOutput, cudaMemcpyHostToDevice);

    // Compute number of blocks and threads per block
    const int threadsPerBlock = maxGatePara;
    const int blocksX = numSignal;
    const int blocksY = numTestcase;
    
    dim3 gridDim(blocksX, blocksY);

    // Run kernel
    double startTimeKernel = CycleTimer::currentSeconds();
    evaluateGates_kernel<<<gridDim, threadsPerBlock, numSignal>>>
                    (device_gates, device_testcase, depth, device_gatePara, device_gateParaSize, device_gateParaStartIdx, numOutput, device_outputId, device_outputVal, device_detected);
    double endTimeKernel = CycleTimer::currentSeconds();

    bool *detected;
    // Copy result from GPU using cudaMemcpy
    cudaMemcpy(detected, device_detected, sizeof(int) * numTestcase * numSignal, cudaMemcpyDeviceToHost);

    // End timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("Overall: %.3f ms", 1000.f * overallDuration);
    double overallDurationKernel = endTimeKernel - startTimeKernel;
    printf("Kernel: %.3f ms", 1000.f * overallDurationKernel);

    // Free memory buffers on the GPU
    cudaFree(device_gates);
    cudaFree(device_testcase);
    cudaFree(device_gatePara);
    cudaFree(device_gateParaSize);
    cudaFree(device_gateParaStartIdx);
    cudaFree(device_outputId);
    cudaFree(device_outputVal);
    cudaFree(device_detected);

    return detected;
}