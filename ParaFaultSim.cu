#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include "CycleTimer.h"
#include "Circuit.h"

__device__
bool evaluateGate(bool *values, GATETYPE gateType, int *gateInput, int gateInputSize, int gateInputStartIdx) {
    // value "true" is 1, "false" is 0
    bool and_gate = true; // AND & NAND gate result
    bool or_gate = false; // OR & NOR gate result
    bool xor_gate = values[gateInput[gateInputStartIdx]]; // XOR & XNOR gate result
    for (size_t i = 0; i < gateInputSize; i++) {
        if (!values[gateInput[gateInputStartIdx + i]]) {
            and_gate = false; // if any value is zero, result is zero
        } else {
            or_gate = true; // if any value is one, result is one
        }
        if (i != 0) xor_gate = xor_gate ^ values[gateInput[gateInputStartIdx + i]];
    }
    if (gateType == BUFF) {
        return values[gateInput[gateInputStartIdx]];
    } else if (gateType == NOT) {
        return !values[gateInput[gateInputStartIdx]];
    } else if (gateType == AND) {
        return and_gate;
    } else if (gateType == NAND) {
        return !and_gate;
    } else if (gateType == OR) {
        return or_gate;
    } else if (gateType == NOR) {
        return !or_gate;
    } else if (gateType == XOR) {
        return xor_gate;
    } else if (gateType == XNOR) {
        return !xor_gate;
    }
}

__global__ void
evaluateGates_GatePara_kernel(GATETYPE *gateType, int *gateInput, int *gateInputSize, int *gateInputStartIdx,
                     bool *testcase, int depth, int *gatePara, int *gateParaSize, int *gateParaStartIdx,
                     int numOutput, int *outputId, bool *outputVal, bool *detected) {
    int numSignal = gridDim.x;
    // int numTestcase = gridDim.y;
    int gateIdx = threadIdx.x; 
    int testcaseIdx = blockIdx.y;
    int faultIdx = blockIdx.x;

    extern __shared__ bool values[]; // Values shared among threads, per fault per testcase
    for (int i = 0; i < depth; i++) {
        // Only gateParaSize[i] number of threads are processed in parallel at once
        if (gateIdx < gateParaSize[i]) {
            int gateId = gatePara[gateParaStartIdx[i] + gateIdx]; // signalID
            // Assign testcase to input values
            if (i == 0) {
                values[gateId] = testcase[testcaseIdx * gateParaSize[0] + gateIdx];
                // Input faults
                if (gateId == faultIdx) {
                    values[gateId] = !values[gateId];
                }
            }
            // Evaluate gates
            else {
                values[gateId] = evaluateGate(values, gateType[gateId], gateInput, gateInputSize[gateId], gateInputStartIdx[gateId]);
                // Signal faults
                if (gateId == faultIdx) {
                    values[gateId] = !values[gateId];
                }
            }
        }
        __syncthreads(); // Sync all threads between gatePara
    }
    
    // Save output values
    if (gateIdx == 0) {
        for (int i = 0; i < numOutput; i++) {
            // Fault can be detected if any faulty circuit output values are different from good circuit output values
            detected[testcaseIdx * numSignal + faultIdx] |= (values[outputId[i]] != outputVal[testcaseIdx * numOutput + i]);
        }
    }
}

bool *
ParaFaultSim_GatePara(int numSignal, int numInput, GATETYPE *gateType, int numGateInput, int *gateInput, int *gateInputSize, int *gateInputStartIdx, int numTestcase, bool *testcase, int depth, int maxGatePara, int *gatePara, int *gateParaSize, int *gateParaStartIdx, int numOutput, int *outputId, bool *outputVal) {

    // Start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    GATETYPE *device_gateType; // 1D gateType[signalID]
    int *device_gateInput; // 2D gateInput[gateID][inputID]
    int *device_gateInputSize; // 1D gateInputSize[signalID]
    int *device_gateInputStartIdx; // 1D gateInputStartIdx[signalID]
    bool *device_testcase; // 2D test[testID][inputID]
    int *device_gatePara; // 1D gatePara[signalID]
    int *device_gateParaSize; // 1D gateParaSize[depth]
    int *device_gateParaStartIdx; // 1D gateParaStartIdx[depth]
    int *device_outputId; // 1D signalID[outputID]
    bool *device_outputVal; // correct output values, 2D outputVal[testID][outputID]
    bool *device_detected; // 2D detected[testID][faultID]

    // Allocate device memory buffers on the GPU using cudaMalloc
    cudaMalloc(&device_gateType, sizeof(GATETYPE) * numSignal);
    cudaMalloc(&device_gateInput, sizeof(int) * numGateInput);
    cudaMalloc(&device_gateInputSize, sizeof(int) * numSignal);
    cudaMalloc(&device_gateInputStartIdx, sizeof(int) * numSignal);
    cudaMalloc(&device_testcase, sizeof(bool) * numTestcase * numInput);
    cudaMalloc(&device_gatePara, sizeof(int) * numSignal);
    cudaMalloc(&device_gateParaSize, sizeof(int) * depth);
    cudaMalloc(&device_gateParaStartIdx, sizeof(int) * depth);
    cudaMalloc(&device_outputId, sizeof(int) * numOutput);
    cudaMalloc(&device_outputVal, sizeof(int) * numTestcase * numOutput);
    cudaMalloc(&device_detected, sizeof(bool) * numTestcase * numSignal);

    // Copy input arrays to the GPU using cudaMemcpy
    cudaMemcpy(device_gateType, gateType, sizeof(GATETYPE) * numSignal, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gateInput, gateInput, sizeof(int) * numGateInput, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gateInputSize, gateInputSize, sizeof(int) * numSignal, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gateInputStartIdx, gateInputStartIdx, sizeof(int) * numSignal, cudaMemcpyHostToDevice);
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

    // End timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    // Run kernel
    double startTimeKernel = CycleTimer::currentSeconds();
    evaluateGates_GatePara_kernel<<<gridDim, threadsPerBlock, sizeof(bool)*numSignal>>>
                    (device_gateType, device_gateInput, device_gateInputSize, device_gateInputStartIdx,
                     device_testcase, depth, device_gatePara, device_gateParaSize, device_gateParaStartIdx, numOutput, device_outputId, device_outputVal, device_detected);
    double endTimeKernel = CycleTimer::currentSeconds();

    bool *detected = new bool[numTestcase * numSignal];
    // Copy result from GPU using cudaMemcpy
    cudaMemcpy(detected, device_detected, sizeof(bool) * numTestcase * numSignal, cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("CUDA mem alloc & copy: %.3f ms\n", 1000.f * overallDuration);
    double overallDurationKernel = endTimeKernel - startTimeKernel;
    printf("Kernel: %.3f ms\n", 1000.f * overallDurationKernel);

    // Free memory buffers on the GPU
    cudaFree(device_gateType);
    cudaFree(device_gateInput);
    cudaFree(device_gateInputSize);
    cudaFree(device_gateInputStartIdx);
    cudaFree(device_testcase);
    cudaFree(device_gatePara);
    cudaFree(device_gateParaSize);
    cudaFree(device_gateParaStartIdx);
    cudaFree(device_outputId);
    cudaFree(device_outputVal);
    cudaFree(device_detected);

    return detected;
}

__global__ void
evaluateGates_TestcasePara_kernel(int numSignal, GATETYPE *gateType, int *gateInput, int *gateInputSize, int *gateInputStartIdx,
                     bool *testcase, int *gatePara, int numInput, int numOutput, int *outputId, bool *outputVal, bool *detected) {
    int partitionIdx = blockIdx.x;
    int faultId = blockDim.x * partitionIdx + threadIdx.x; // Global fault id within a testcase
    int faultIdx = threadIdx.x; // Local partition fault index
    int testcaseIdx = blockIdx.y;

    extern __shared__ bool values[]; // numFault * numSignal
    if (faultId < numSignal) {
        for (int i = 0; i < numSignal; i++) {
            int gateId = gatePara[i]; // signalID
            int valueIdx = faultIdx * numSignal + gateId; 
            // Assign testcase to input values
            if (gateId < numInput) {
                values[valueIdx] = testcase[testcaseIdx * numInput + gateId];
                // Input faults
                if (gateId == faultId) {
                    values[valueIdx] = !values[valueIdx];
                }
            }
            // Evaluate gates
            else {
                values[valueIdx] = evaluateGate((values + faultIdx * numSignal), gateType[gateId], gateInput, gateInputSize[gateId], gateInputStartIdx[gateId]);
                // Signal faults
                if (gateId == faultId) {
                    values[valueIdx] = !values[valueIdx];
                }
            }
        }
    
        // Save output values
        for (int i = 0; i < numOutput; i++) {
            // Fault can be detected if any faulty circuit output values are different from good circuit output values
            detected[testcaseIdx * numSignal + faultId] |= (values[faultIdx * numSignal + outputId[i]] != outputVal[testcaseIdx * numOutput + i]);
        }
    }
}

bool *
ParaFaultSim_TestcasePara(int numSignal, int numInput, GATETYPE *gateType, int numGateInput, int *gateInput, int *gateInputSize, int *gateInputStartIdx, int numTestcase, bool *testcase, int *gatePara, int numOutput, int *outputId, bool *outputVal) {

    // Start timing after allocation of device memory
    double startTime = CycleTimer::currentSeconds();

    GATETYPE *device_gateType; // 1D gateType[signalID]
    int *device_gateInput; // 2D gateInput[gateID][inputID]
    int *device_gateInputSize; // 1D gateInputSize[signalID]
    int *device_gateInputStartIdx; // 1D gateInputStartIdx[signalID]
    bool *device_testcase; // 2D test[testID][inputID]
    int *device_gatePara; // 1D gatePara[signalID]
    int *device_outputId; // 1D signalID[outputID]
    bool *device_outputVal; // correct output values, 2D outputVal[testID][outputID]
    bool *device_detected; // 2D detected[testID][faultID]

    // Allocate device memory buffers on the GPU using cudaMalloc
    cudaMalloc(&device_gateType, sizeof(GATETYPE) * numSignal);
    cudaMalloc(&device_gateInput, sizeof(int) * numGateInput);
    cudaMalloc(&device_gateInputSize, sizeof(int) * numSignal);
    cudaMalloc(&device_gateInputStartIdx, sizeof(int) * numSignal);
    cudaMalloc(&device_testcase, sizeof(bool) * numTestcase * numInput);
    cudaMalloc(&device_gatePara, sizeof(int) * numSignal);
    cudaMalloc(&device_outputId, sizeof(int) * numOutput);
    cudaMalloc(&device_outputVal, sizeof(int) * numTestcase * numOutput);
    cudaMalloc(&device_detected, sizeof(bool) * numTestcase * numSignal);

    // Copy input arrays to the GPU using cudaMemcpy
    cudaMemcpy(device_gateType, gateType, sizeof(GATETYPE) * numSignal, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gateInput, gateInput, sizeof(int) * numGateInput, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gateInputSize, gateInputSize, sizeof(int) * numSignal, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gateInputStartIdx, gateInputStartIdx, sizeof(int) * numSignal, cudaMemcpyHostToDevice);
    cudaMemcpy(device_testcase, testcase, sizeof(bool) * numTestcase * numInput, cudaMemcpyHostToDevice);
    cudaMemcpy(device_gatePara, gatePara, sizeof(int) * numSignal, cudaMemcpyHostToDevice);
    cudaMemcpy(device_outputId, outputId, sizeof(int) * numOutput, cudaMemcpyHostToDevice);
    cudaMemcpy(device_outputVal, outputVal, sizeof(int) * numTestcase * numOutput, cudaMemcpyHostToDevice);

    // Compute number of blocks and threads per block
    const int partition = 256;
    const int threadsPerBlock = (numSignal + partition - 1) / partition;
    dim3 gridDim (partition, numTestcase);

    // End timing after result has been copied back into host memory
    double endTime = CycleTimer::currentSeconds();

    // Run kernel
    double startTimeKernel = CycleTimer::currentSeconds();
    evaluateGates_TestcasePara_kernel<<<gridDim, threadsPerBlock, sizeof(bool)*((numSignal+partition-1)/partition) * numSignal>>>
                    (numSignal, device_gateType, device_gateInput, device_gateInputSize, device_gateInputStartIdx,
                     device_testcase, device_gatePara, numInput, numOutput, device_outputId, device_outputVal, device_detected);
    double endTimeKernel = CycleTimer::currentSeconds();

    bool *detected = new bool[numTestcase * numSignal];
    // Copy result from GPU using cudaMemcpy
    cudaMemcpy(detected, device_detected, sizeof(bool) * numTestcase * numSignal, cudaMemcpyDeviceToHost);

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }

    double overallDuration = endTime - startTime;
    printf("CUDA mem alloc & copy: %.3f ms\n", 1000.f * overallDuration);
    double overallDurationKernel = endTimeKernel - startTimeKernel;
    printf("Kernel: %.3f ms\n", 1000.f * overallDurationKernel);

    // Free memory buffers on the GPU
    cudaFree(device_gateType);
    cudaFree(device_gateInput);
    cudaFree(device_gateInputSize);
    cudaFree(device_gateInputStartIdx);
    cudaFree(device_testcase);
    cudaFree(device_gatePara);
    cudaFree(device_outputId);
    cudaFree(device_outputVal);
    cudaFree(device_detected);

    return detected;
}